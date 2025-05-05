import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    filename="dataset_creation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

def define_tool_schemas():
    """Return the formal tool schema for SWE-agent tools, including str_replace_editor."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Runs the given command directly in bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute."}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "goto",
                "description": "Moves the window to show <line_number>",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_number": {"type": "integer", "description": "The line number to move the window to"}
                    },
                    "required": ["line_number"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "open",
                "description": "Opens the file at the given path in the editor. If line_number is provided, the window will move to include that line",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to open"},
                        "line_number": {"type": "integer", "description": "The line number to move the window to (if not provided, starts at the top)"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create",
                "description": "Creates and opens a new file with the given name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "The name of the file to create"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scroll_up",
                "description": "Moves the window up {WINDOW} lines",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scroll_down",
                "description": "Moves the window down {WINDOW} lines",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_file",
                "description": "Finds all files with the given name or pattern in dir. If dir is not provided, searches in the current directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string", "description": "The name or pattern to search for (e.g., *.py)"},
                        "dir": {"type": "string", "description": "The directory to search in (default: current)"}
                    },
                    "required": ["file_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_dir",
                "description": "Searches for search_term in all files in dir. If dir is not provided, searches in the current directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string", "description": "The term to search for"},
                        "dir": {"type": "string", "description": "The directory to search in (default: current)"}
                    },
                    "required": ["search_term"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_file",
                "description": "Searches for search_term in file. If file is not provided, searches in the current open file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string", "description": "The term to search for"},
                        "file": {"type": "string", "description": "The file to search in (default: current)"}
                    },
                    "required": ["search_term"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replaces first occurrence of <search> with <replace> in the currently displayed lines. If replace-all is True, replaces all occurrences.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string", "description": "The text to search for"},
                        "replace": {"type": "string", "description": "The text to replace with"},
                        "replace-all": {"type": "boolean", "description": "Replace all occurrences if True"}
                    },
                    "required": ["search", "replace"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert",
                "description": "Inserts <text> at the end of the current file or after <line> if specified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to insert"},
                        "line": {"type": "integer", "description": "The line number to insert after"}
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submits the current file",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": (
                    "Custom editing tool for viewing, creating, and editing files. State is persistent across command calls. "
                    "Commands include: view (displays file with line numbers or directory listing), create (creates a new file), "
                    "str_replace (replaces a string), insert (inserts a string after a line), and undo_edit (reverts last edit). "
                    "For str_replace, old_str must match exactly and be unique; otherwise, no replacement occurs. "
                    "Long outputs are truncated with '<response clipped>'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run. Options: view, create, str_replace, insert, undo_edit.",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"]
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file or directory, e.g., /testbed/file.py or /testbed."
                        },
                        "file_text": {
                            "type": "string",
                            "description": "Content of the file to be created. Required for create command."
                        },
                        "old_str": {
                            "type": "string",
                            "description": (
                                "String to replace in the file. Required for str_replace command. "
                                "Must match exactly one or more consecutive lines, including whitespaces, and be unique."
                            )
                        },
                        "new_str": {
                            "type": "string",
                            "description": (
                                "Replacement string for str_replace or content to insert for insert command. "
                                "Required for insert; optional for str_replace (if omitted, no string is added)."
                            )
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "Line number after which to insert new_str. Required for insert command."
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Line number range for view command when path is a file, e.g., [11, 12] shows lines 11-12. "
                                "Indexing starts at 1. Use [start_line, -1] to show from start_line to end. Optional."
                            )
                        }
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False
                }
            }
        }
    ]

def process_conversation(messages):
    """Process a conversation to simplify tool responses and map to Qwen format."""
    if not messages or not isinstance(messages, list):
        logger.warning(f"Invalid messages: type={type(messages)}, content={messages}")
        return []

    tool_schemas = define_tool_schemas()
    valid_tool_names = {tool["function"]["name"] for tool in tool_schemas}
    processed_messages = []

    # Add system message
    has_system = False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            has_system = True
            system_msg = msg.copy()
            if "tools" not in system_msg:
                system_msg["tools"] = tool_schemas
            processed_messages.append(system_msg)
            break
    if not has_system:
        processed_messages.append({
            "role": "system",
            "content": "You are a coding assistant with access to tools.",
            "tools": tool_schemas
        })

    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            logger.warning(f"Skipping invalid message at index {i}: {msg}")
            i += 1
            continue
        role = msg.get("role", "missing")

        if role == "user":
            processed_messages.append(msg)
            i += 1

        elif role == "assistant" and "tool_calls" in msg:
            # Assistant message with tool calls comes first
            assistant_msg = {
                "role": "assistant",
                "content": msg.get("content", ""),
                "function_call": []
            }
            tool_calls = msg.get("tool_calls", [])
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_name = tool_call.get("name", "")
                if tool_name not in valid_tool_names:
                    logger.warning(f"Invalid tool name '{tool_name}'")
                function_call = {
                    "name": tool_name,
                    "arguments": tool_call.get("arguments", {})
                }
                if "id" in tool_call:
                    function_call["tool_call_id"] = tool_call["id"]
                assistant_msg["function_call"].append(function_call)
            processed_messages.append(assistant_msg)
            i += 1

            # Then process tool responses
            while i < len(messages):
                next_msg = messages[i]
                if not isinstance(next_msg, dict) or next_msg.get("role") != "tool":
                    break
                tool_name = next_msg.get("name", "")
                function_msg = {
                    "role": "function",
                    "name": tool_name,
                    "content": next_msg.get("content", ""),
                    "tool_call_id": next_msg.get("tool_call_id", "")
                }
                processed_messages.append(function_msg)
                i += 1

        elif role != "system":
            processed_messages.append(msg)
            i += 1
        else:
            i += 1

    return processed_messages

def create_simplified_dataset(input_path, output_path, max_lines=10000):
    """Create a simplified dataset from SWE-agent logs."""
    total_examples = 0
    processed_examples = 0
    invalid_lines = []

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, 1):
            if i > max_lines:
                logger.info(f"Stopped after {max_lines} lines")
                break
            try:
                example = json.loads(line.strip())
                total_examples += 1
                messages = example.get("messages", [])
                if not isinstance(messages, list):
                    logger.warning(f"Line {i}: 'messages' not a list: {messages}")
                    invalid_lines.append(i)
                    continue

                simplified = process_conversation(messages)
                if simplified:
                    outfile.write(json.dumps({"messages": simplified}) + "\n")
                    processed_examples += 1
                    logger.info(f"Line {i}: Processed {len(messages)} -> {len(simplified)} messages")
                else:
                    logger.warning(f"Line {i}: No output messages")
                    invalid_lines.append(i)

            except json.JSONDecodeError as e:
                logger.error(f"Line {i}: JSON error: {e}")
                invalid_lines.append(i)
            except Exception as e:
                logger.error(f"Line {i}: Unexpected error: {e}")
                invalid_lines.append(i)

    # Summary
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Processed examples: {processed_examples}")
    logger.info(f"Invalid lines: {len(invalid_lines)}")
    print(f"Processed {processed_examples}/{total_examples} examples.")
    print(f"Output saved to {output_path}. See dataset_creation.log for details.")
    if invalid_lines:
        print(f"Warning: {len(invalid_lines)} lines failed. Check log.")

if __name__ == "__main__":
    input_path = "full_swe_agent_logs_resolved.jsonl"
    output_path = "simplified_swe_agent_logs.jsonl"
    create_simplified_dataset(input_path, output_path)
