import json
import random
import logging
from collections import Counter
from pathlib import Path
from itertools import groupby

# Set up logging to file with clean output (no timestamp or INFO)
log_file = "jsonl_validation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s"  # Remove timestamp and level
)
logger = logging.getLogger()

def summarize_role_key_sequence(messages):
    """
    Summarize a list of messages into consecutive role-key set counts.
    E.g., [{'role': 'assistant', keys: ['role', 'content']}, {'role': 'tool', keys: ['role', 'name']}] 
    -> [(['assistant', ['role', 'content']], 1), (['tool', ['role', 'name']], 1)]
    """
    role_key_pairs = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "missing")
            keys = sorted(msg.keys())
            role_key_pairs.append((role, keys))
        else:
            role_key_pairs.append(("invalid", ["invalid"]))
    return [((role, keys), len(list(group))) for (role, keys), group in groupby(role_key_pairs)]

def validate_and_analyze_jsonl(file_path, sample_size=5, max_lines=10000):
    invalid_lines = []
    tool_def_count = 0
    function_call_count = 0
    function_response_count = 0
    total_examples = 0
    samples_metadata = []
    role_counts = Counter()
    message_keys = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > max_lines:  # Limit analysis to avoid excessive processing
                    break
                try:
                    example = json.loads(line.strip())
                    total_examples += 1

                    # Analyze messages for tool-calling structure and metadata
                    if "messages" in example:
                        messages = example.get("messages", [])
                        if not isinstance(messages, list):
                            logger.warning(f"Line {i}: 'messages' is not a list: {type(messages)}")
                            continue

                        # Extract metadata and role-key sequence
                        sample_info = {
                            "line": i,
                            "message_count": len(messages),
                            "roles": [],
                            "has_tools": False,
                            "has_function_call": False,
                            "has_function_role": False,
                            "keys_per_message": []
                        }

                        for msg in messages:
                            if not isinstance(msg, dict):
                                sample_info["keys_per_message"].append({"error": "not a dict"})
                                sample_info["roles"].append("invalid")
                                continue
                            role = msg.get("role", "missing")
                            sample_info["roles"].append(role)
                            role_counts[role] += 1
                            keys = list(msg.keys())
                            sample_info["keys_per_message"].append({role: keys})
                            message_keys.update(keys)
                            if role == "system" and "tools" in msg:
                                sample_info["has_tools"] = True
                                tool_def_count += 1
                            if role == "assistant" and "function_call" in msg:
                                sample_info["has_function_call"] = True
                                function_call_count += 1
                            if role == "function":
                                sample_info["has_function_role"] = True
                                function_response_count += 1

                        # Summarize role-key sequence and log if sample is selected
                        if messages and len(samples_metadata) < sample_size:
                            summarized = summarize_role_key_sequence(messages)
                            logger.info(f"Sample {len(samples_metadata) + 1} (Line {i}) Role-Key Sequence:")
                            for (role, keys), count in summarized:
                                logger.info(f'["{role}", {keys}], {count}')
                            # Log full tool_calls data for messages with tool_calls
                            logger.info(f"Tool Calls in Sample {len(samples_metadata) + 1} (Line {i}):")
                            tool_calls_found = False
                            for msg in messages:
                                if isinstance(msg, dict) and "tool_calls" in msg:
                                    tool_calls = msg.get("tool_calls", [])
                                    logger.info(f"Message (role: {msg.get('role', 'missing')}):")
                                    logger.info(json.dumps(tool_calls, indent=2))
                                    tool_calls_found = True
                            if not tool_calls_found:
                                logger.info("No tool_calls found in this sample.")
                            samples_metadata.append(sample_info)

                except json.JSONDecodeError as e:
                    invalid_lines.append((i, str(e)))
                except Exception as e:
                    logger.error(f"Unexpected error at line {i}: {e}")

        # Log metadata for samples
        logger.info(f"Collected metadata for {len(samples_metadata)} sample entries:")
        for i, meta in enumerate(samples_metadata, 1):
            logger.info(f"Sample {i} (Line {meta['line']}):\n{json.dumps(meta, indent=2)}")

        # Log invalid lines
        if invalid_lines:
            logger.info(f"Found {len(invalid_lines)} invalid JSON lines:")
            for line_num, error in invalid_lines[:10]:  # Log first 10
                logger.info(f"Line {line_num}: Error: {error}")
            if len(invalid_lines) > 10:
                logger.info(f"Additional {len(invalid_lines) - 10} invalid lines omitted.")

        # Log role counts and message keys
        logger.info(f"Role counts across dataset:\n{json.dumps(dict(role_counts), indent=2)}")
        logger.info(f"Unique message keys:\n{json.dumps(sorted(list(message_keys)), indent=2)}")

        # Print minimal insights to console
        print(f"Validation Summary (details in {log_file}):")
        print(f"Total examples analyzed: {total_examples}")
        print(f"Invalid JSON lines: {len(invalid_lines)}")
        print(f"Samples with metadata logged: {len(samples_metadata)}")
        print(f"Examples with tool definitions (system.tools): {tool_def_count}")
        print(f"Examples with function calls (assistant.function_call): {function_call_count}")
        print(f"Examples with function responses (role=function): {function_response_count}")
        print(f"Message roles found: {list(role_counts.keys())}")
        print(f"Unique message keys: {sorted(list(message_keys))}")

        return invalid_lines, samples_metadata, tool_def_count, function_call_count, function_response_count

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        logger.error(f"File {file_path} not found")
        return [], [], 0, 0, 0
    except Exception as e:
        print(f"Error: Failed to process file: {e}")
        logger.error(f"Failed to process file: {e}")
        return [], [], 0, 0, 0

if __name__ == "__main__":
    file_path = "/home/srkashy1/SWE-agent/full_swe_agent_logs_resolved.jsonl"
    invalid_lines, samples_metadata, tool_def_count, function_call_count, function_response_count = validate_and_analyze_jsonl(file_path)
    if invalid_lines:
        print("Warning: Invalid JSON lines detected. Check jsonl_validation.log for details.")
    if not (tool_def_count > 0 and function_call_count > 0):
        print("Warning: Dataset may lack tool-calling structures. Check jsonl_validation.log for sample metadata.")