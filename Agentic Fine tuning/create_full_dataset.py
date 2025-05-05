import json
import os
import glob
import argparse
import traceback # For detailed error reporting

# --- Core processing functions (adapted from create_single_training_data.py) ---

def format_qwen_tool_call(tool_call):
    """Formats a single tool call for Qwen models."""
    if not tool_call or 'function' not in tool_call:
        return None
    func = tool_call['function']
    args_str = func.get('arguments', '{}')
    return {
        "name": func.get('name'),
        "arguments": args_str,
        "id": tool_call.get('id') # Include the original ID if available
    }

def format_tool_response(tool_call_id, function_name, observation_text):
    """Formats the tool's observation/response."""
    response = {
        "role": "tool", # Using 'tool' as it's common now
        "content": observation_text,
        "tool_call_id": tool_call_id
    }
    # Add 'name' if available, expected by some processing templates
    if function_name:
        response["name"] = function_name
    return response

def process_single_traj(traj_file_path, system_prompt):
    """
    Processes a single .traj file and returns the 'messages' list.
    Accepts system_prompt as argument to avoid reloading tools repeatedly.

    Args:
        traj_file_path (str): Path to the specific .traj file.
        system_prompt (str): The pre-formatted system prompt including tool definitions.

    Returns:
        list or None: The list of messages for the trajectory, or None if an error occurs.
    """
    messages = []
    try:
        with open(traj_file_path, 'r') as f:
            data = json.load(f)

        if "trajectory" not in data or not isinstance(data["trajectory"], list):
            print(f"Warning: Skipping {traj_file_path} - 'trajectory' key missing or invalid.")
            return None

        # Start with the provided system prompt
        messages.append({"role": "system", "content": system_prompt})

        call_id_to_function_name = {}
        last_role = None

        for step_index, step in enumerate(data["trajectory"]):
            step_messages = step.get("messages", [])

            # --- User Message ---
            user_message = next((msg for msg in step_messages if msg.get("role") == "user"), None)
            if user_message:
                user_content = ""
                if isinstance(user_message.get("content"), list):
                     text_parts = [part.get("text", "") for part in user_message["content"] if isinstance(part, dict) and "text" in part]
                     user_content = "\n".join(filter(None, text_parts))
                elif isinstance(user_message.get("content"), str):
                     user_content = user_message["content"]

                if user_content and (not messages or messages[-1].get("content") != user_content):
                    messages.append({"role": "user", "content": user_content})
                    last_role = "user"

            # --- Assistant Message & Tool Calls ---
            assistant_content = step.get("response")
            assistant_actions = [msg for msg in step_messages if msg.get("agent") == "main" and msg.get("message_type") == "action" and msg.get("role") == "assistant"]
            formatted_tool_calls = []
            if assistant_actions:
                for action_msg in assistant_actions:
                     raw_calls = action_msg.get("tool_calls", [])
                     for raw_call in raw_calls:
                         fmt_call = format_qwen_tool_call(raw_call)
                         if fmt_call:
                             formatted_tool_calls.append(fmt_call)
                             if fmt_call.get("id") and fmt_call.get("name"):
                                 call_id_to_function_name[fmt_call["id"]] = fmt_call["name"]

            # Construct the assistant message
            assistant_msg = None
            if assistant_content or formatted_tool_calls:
                assistant_msg = {"role": "assistant"}
                if assistant_content:
                    assistant_msg["content"] = assistant_content
                if formatted_tool_calls:
                    assistant_msg["tool_calls"] = formatted_tool_calls # Qwen expects tool_calls list

                if not messages or messages[-1] != assistant_msg:
                     messages.append(assistant_msg)
                     last_role = "assistant"

            # --- Tool Observation/Response ---
            tool_observations = [msg for msg in step_messages if msg.get("agent") == "main" and msg.get("message_type") == "observation" and msg.get("role") == "tool"]
            if tool_observations:
                for obs_msg in tool_observations:
                     obs_content_list = obs_msg.get("content", [])
                     tool_call_ids = obs_msg.get("tool_call_ids", [])
                     obs_text = ""
                     if isinstance(obs_content_list, list):
                          text_parts = [part.get("text", "") for part in obs_content_list if isinstance(part, dict) and "text" in part]
                          obs_text = "\n".join(filter(None, text_parts)).replace("OBSERVATION:", "").strip()
                     elif isinstance(obs_content_list, str):
                          obs_text = obs_content_list.replace("OBSERVATION:", "").strip()

                     if obs_text and tool_call_ids:
                          for tool_call_id in tool_call_ids:
                              if tool_call_id:
                                   function_name = call_id_to_function_name.get(tool_call_id)
                                   # if not function_name: # Optional: Add warning if name missing
                                   #    print(f"Warning: Could not find function name for tool_call_id {tool_call_id} in {traj_file_path}")
                                   fmt_response = format_tool_response(tool_call_id, function_name, obs_text)
                                   messages.append(fmt_response)
                                   last_role = "tool"
                     # else: # Optional: Add warning for observations without text or IDs
                     #    print(f"Warning: Tool observation missing text or tool_call_ids in step {step_index} of {traj_file_path}")


        # Final check: ensure conversation doesn't end with assistant action without response
        if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
             # print(f"Warning: Trajectory {traj_file_path} ends with an assistant action.")
             pass # Keep as is for now

        return messages

    except json.JSONDecodeError:
        print(f"Error decoding JSON from {traj_file_path}. Skipping.")
        traceback.print_exc()
        return None
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at {traj_file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing {traj_file_path}: {e}. Skipping.")
        traceback.print_exc()
        return None

# --- Main Script Logic ---

def create_jsonl_from_directory(traj_root_dir, tool_definitions_path, output_path):
    """
    Scans a directory, processes resolved .traj files, and creates a JSONL file.

    Args:
        traj_root_dir (str): The root directory containing the trajectory folders.
        tool_definitions_path (str): Path to the JSON file with formal tool definitions.
        output_path (str): Path to the output JSONL file.
    """

    print(f"Loading tool definitions from: {tool_definitions_path}")
    try:
        with open(tool_definitions_path, 'r') as f:
            formal_tools = json.load(f)
            tools_string = json.dumps(formal_tools.get("tools", []), indent=2)
            # Define the system prompt once
            system_prompt = (
                 "You are a helpful assistant. The following functions are available for you to call:\n"
                 f"{tools_string}\n"
                 "To call a function, respond - immediately and only - with a JSON object matching the following schema:\n"
                 "{\n"
                 "  \"name\": \"function_name\",\n"
                 "  \"arguments\": {\n"
                 "    \"arg_1\": \"value_1\",\n"
                 "    \"arg_2\": \"value_2\",\n"
                 "    ...\n"
                 "  }\n"
                 "}"
            )
            print("Tool definitions loaded successfully.")
    except Exception as e:
        print(f"Error loading tool definitions: {e}")
        return

    print(f"Scanning for trajectories in: {traj_root_dir}")
    # Find all report.json files recursively
    report_files = glob.glob(os.path.join(traj_root_dir, '**', 'report.json'), recursive=True)

    print(f"Found {len(report_files)} potential trajectory reports.")
    if not report_files:
        print("No report.json files found. Please check the directory path and structure.")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as outfile:
        # Iterate through found report files (using manual loop for clarity without tqdm)
        total_reports = len(report_files)
        for i, report_path in enumerate(report_files):
            if (i + 1) % 50 == 0: # Print progress update every 50 files
                print(f"  Processing report {i+1}/{total_reports}...")

            instance_dir = os.path.dirname(report_path)
            instance_id = os.path.basename(instance_dir) # e.g., astropy__astropy-6938

            # Construct the expected .traj file path
            traj_file_path = os.path.join(instance_dir, f"{instance_id}.traj")

            try:
                with open(report_path, 'r') as f_report:
                    report_data = json.load(f_report)

                # Check the 'resolved' status - dynamically access using instance_id key
                if instance_id not in report_data:
                     print(f"Warning: Instance ID '{instance_id}' not found as key in {report_path}. Skipping.")
                     skipped_count += 1
                     continue

                is_resolved = report_data[instance_id].get("resolved", False)

                if is_resolved:
                    # Check if the corresponding .traj file exists
                    if not os.path.exists(traj_file_path):
                        print(f"Warning: report.json indicates resolved, but .traj file not found at {traj_file_path}. Skipping.")
                        skipped_count += 1
                        continue

                    # Process the trajectory
                    messages = process_single_traj(traj_file_path, system_prompt)

                    if messages and len(messages) > 1: # Ensure more than just system prompt
                        outfile.write(json.dumps({"messages": messages}) + '\n')
                        processed_count += 1
                    elif messages is None: # Check if process_single_traj returned None due to error
                        error_count += 1
                    else: # Trajectory resulted in only system prompt or empty
                         skipped_count += 1
                         # print(f"Info: Trajectory {traj_file_path} resulted in empty/system-only messages. Skipping.")

                else:
                    # print(f"Info: Trajectory {instance_id} is not resolved. Skipping.")
                    skipped_count += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON from report file {report_path}. Skipping associated trajectory.")
                error_count += 1
            except Exception as e:
                print(f"Unexpected error processing report {report_path} or trajectory {traj_file_path}: {e}")
                traceback.print_exc()
                error_count += 1

    print(f"\nProcessing complete.")
    print(f"  Successfully processed and wrote: {processed_count} trajectories.")
    print(f"  Skipped (not resolved or invalid): {skipped_count} trajectories.")
    print(f"  Encountered errors: {error_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert resolved SWE-Agent .traj files from a directory to JSONL for SFT.")
    parser.add_argument("traj_root_dir", help="Root directory containing the trajectory subfolders (e.g., './swe-agent-claude-3-7')")
    parser.add_argument("tool_definitions", help="Path to the formal_tool_definitions.json file.")
    parser.add_argument("output_file", help="Path for the output JSONL file (e.g., './resolved_swe_agent_logs.jsonl')")

    args = parser.parse_args()

    # Use absolute paths for clarity
    traj_directory = os.path.abspath(args.traj_root_dir)
    tool_definitions_file = os.path.abspath(args.tool_definitions)
    output_jsonl_path = os.path.abspath(args.output_file)

    create_jsonl_from_directory(traj_directory, tool_definitions_file, output_jsonl_path)
