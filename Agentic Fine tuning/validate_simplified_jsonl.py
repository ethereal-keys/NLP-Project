import json
import logging
from collections import Counter
from pathlib import Path
from itertools import groupby

# Set up logging
log_file = "simplified_jsonl_validation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger()

def define_tool_schemas():
    """Return the formal tool schema for validation."""
    return {
        "bash": {"parameters": {"command": "string"}, "required": ["command"]},
        "goto": {"parameters": {"line_number": "integer"}, "required": ["line_number"]},
        "open": {"parameters": {"path": "string", "line_number": "integer"}, "required": ["path"]},
        "create": {"parameters": {"filename": "string"}, "required": ["filename"]},
        "scroll_up": {"parameters": {}, "required": []},
        "scroll_down": {"parameters": {}, "required": []},
        "find_file": {"parameters": {"file_name": "string", "dir": "string"}, "required": ["file_name"]},
        "search_dir": {"parameters": {"search_term": "string", "dir": "string"}, "required": ["search_term"]},
        "search_file": {"parameters": {"search_term": "string", "file": "string"}, "required": ["search_term"]},
        "edit": {"parameters": {"search": "string", "replace": "string", "replace-all": "boolean"}, "required": ["search", "replace"]},
        "insert": {"parameters": {"text": "string", "line": "integer"}, "required": ["text"]},
        "submit": {"parameters": {}, "required": []},
        "str_replace_editor": {
            "parameters": {
                "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
                "path": "string",
                "file_text": "string",
                "old_str": "string",
                "new_str": "string",
                "insert_line": "integer",
                "view_range": {"type": "array", "items": "integer"}
            },
            "required": ["command", "path"]
        }
    }

def summarize_role_key_sequence(messages):
    """Summarize consecutive role-key set counts."""
    role_key_pairs = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(f"Invalid message at index {i}: type={type(msg)}, content={msg}")
            role_key_pairs.append(("invalid", [str(type(msg))]))
            continue
        role = msg.get("role", "missing")
        keys = sorted(msg.keys())
        role_key_pairs.append((role, keys))
    return [((role, keys), len(list(group))) for (role, keys), group in groupby(role_key_pairs)]

def validate_and_analyze_jsonl(file_path, sample_size=5, max_lines=10000):
    """Validate and analyze the simplified JSONL dataset."""
    invalid_lines = []
    tool_def_count = 0
    function_call_count = 0
    function_response_count = 0
    total_examples = 0
    samples_metadata = []
    role_counts = Counter()
    message_keys = set()
    tool_names = set()
    invalid_tools = set()
    tool_schemas = define_tool_schemas()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > max_lines:
                    logger.info(f"Stopped processing after {max_lines} lines")
                    break
                try:
                    example = json.loads(line.strip())
                    total_examples += 1

                    if "messages" not in example:
                        logger.warning(f"Line {i}: Missing 'messages' key")
                        invalid_lines.append((i, "Missing 'messages' key"))
                        continue
                    messages = example.get("messages", [])
                    if not isinstance(messages, list):
                        logger.warning(f"Line {i}: 'messages' is not a list: type={type(messages)}, content={messages}")
                        invalid_lines.append((i, f"'messages' is not a list: {type(messages)}"))
                        continue

                    sample_info = {
                        "line": i,
                        "message_count": len(messages),
                        "roles": [],
                        "has_tools": False,
                        "has_function_call": False,
                        "has_function_role": False,
                        "keys_per_message": []
                    }

                    for j, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            logger.warning(f"Line {i}, message {j}: Invalid message, not a dict: type={type(msg)}, content={msg}")
                            sample_info["keys_per_message"].append({"error": f"not a dict: {type(msg)}"})
                            sample_info["roles"].append("invalid")
                            role_counts["invalid"] += 1
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
                            function_calls = msg.get("function_call", [])
                            if not isinstance(function_calls, list):
                                logger.warning(f"Line {i}, message {j}: function_call is not a list: type={type(function_calls)}, content={function_calls}")
                                continue
                            for k, call in enumerate(function_calls):
                                if not isinstance(call, dict):
                                    logger.warning(f"Line {i}, message {j}, function_call {k}: Invalid entry: type={type(call)}, content={call}")
                                    continue
                                tool_name = call.get("name", "")
                                if tool_name in tool_schemas:
                                    tool_names.add(tool_name)
                                else:
                                    invalid_tools.add(tool_name)
                        if role == "function":
                            sample_info["has_function_role"] = True
                            function_response_count += 1
                            tool_name = msg.get("name", "")
                            if tool_name in tool_schemas:
                                tool_names.add(tool_name)
                            else:
                                invalid_tools.add(tool_name)

                    if messages and len(samples_metadata) < sample_size:
                        summarized = summarize_role_key_sequence(messages)
                        logger.info(f"Sample {len(samples_metadata) + 1} (Line {i}) Role-Key Sequence:")
                        for (role, keys), count in summarized:
                            logger.info(f'["{role}", {keys}], {count}')
                        logger.info(f"Function Calls in Sample {len(samples_metadata) + 1} (Line {i}):")
                        function_calls_found = False
                        for j, msg in enumerate(messages):
                            if not isinstance(msg, dict):
                                continue
                            if msg.get("role") == "assistant" and "function_call" in msg:
                                logger.info(f"Message {j} (role: assistant):")
                                logger.info(json.dumps(msg.get("function_call", []), indent=2))
                                function_calls_found = True
                            elif msg.get("role") == "function":
                                logger.info(f"Message {j} (role: function, name: {msg.get('name', 'missing')}):")
                                logger.info(json.dumps({"content": msg.get("content"), "tool_call_id": msg.get("tool_call_id")}, indent=2))
                                function_calls_found = True
                        if not function_calls_found:
                            logger.info("No function calls or responses found in this sample.")
                        samples_metadata.append(sample_info)

                except json.JSONDecodeError as e:
                    invalid_lines.append((i, str(e)))
                    logger.error(f"Line {i}: JSON decode error: {e}")
                except Exception as e:
                    invalid_lines.append((i, str(e)))
                    logger.error(f"Unexpected error at line {i}: {e}")

        logger.info(f"Collected metadata for {len(samples_metadata)} sample entries:")
        for i, meta in enumerate(samples_metadata, 1):
            logger.info(f"Sample {i} (Line {meta['line']}):\n{json.dumps(meta, indent=2)}")

        if invalid_lines:
            logger.info(f"Found {len(invalid_lines)} invalid JSON lines or processing errors:")
            for line_num, error in invalid_lines[:10]:
                logger.info(f"Line {line_num}: Error: {error}")
            if len(invalid_lines) > 10:
                logger.info(f"Additional {len(invalid_lines) - 10} invalid lines omitted.")

        logger.info(f"Role counts across dataset:\n{json.dumps(dict(role_counts), indent=2)}")
        logger.info(f"Unique message keys:\n{json.dumps(sorted(list(message_keys)), indent=2)}")
        logger.info(f"Valid tool names found:\n{json.dumps(sorted(list(tool_names)), indent=2)}")
        if invalid_tools:
            logger.info(f"Invalid tool names found:\n{json.dumps(sorted(list(invalid_tools)), indent=2)}")

        print(f"Validation Summary (details in {log_file}):")
        print(f"Total examples analyzed: {total_examples}")
        print(f"Invalid JSON lines or processing errors: {len(invalid_lines)}")
        print(f"Samples with metadata logged: {len(samples_metadata)}")
        print(f"Examples with tool definitions (system.tools): {tool_def_count}")
        print(f"Examples with function calls (assistant.function_call): {function_call_count}")
        print(f"Examples with function responses (role=function): {function_response_count}")
        print(f"Message roles found: {sorted(list(role_counts.keys()))}")
        print(f"Unique message keys: {sorted(list(message_keys))}")
        print(f"Valid tool names: {sorted(list(tool_names))}")
        if invalid_tools:
            print(f"Invalid tool names found: {sorted(list(invalid_tools))}")

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
    file_path = "./cleaned_swe_agent_logs.jsonl"
    invalid_lines, samples_metadata, tool_def_count, function_call_count, function_response_count = validate_and_analyze_jsonl(file_path)
    if invalid_lines:
        print("Warning: Invalid JSON lines or processing errors detected. Check simplified_jsonl_validation.log for details.")
    if not (tool_def_count > 0 and function_call_count > 0):
        print("Warning: Dataset may lack tool-calling structures. Check simplified_jsonl_validation.log for sample metadata.")
