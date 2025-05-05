def get_tool_definitions():
    """Get example tool definitions to include in system message"""
    tools = [
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
                "description": "Moves the window up \{WINDOW\} lines",
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
    return tools

import json
from typing import List, Dict, Any, Optional
import functools
from pprint import pprint
import re

# Import the exact Mistral tokenizer classes referenced in your example
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    ToolMessage,
    SystemMessage
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.tool_calls import Function, Tool, ToolCall, FunctionCall
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {file_path}: {e}")
    return data

def convert_to_mistral_messages(conversation: List[Dict[str, Any]]) -> List:
    """Convert a conversation to Mistral message objects."""
    mistral_messages = []
    
    for entry in conversation:
        role = entry.get('role')
        # print(role)
        content = entry.get('content', '')
        # print(type(content))
        
        # Skip empty content or placeholder content
        # if content in [None, '', '### SOME CONTENT ###']:
        #     continue
            
        if role == 'system':
            # print("hi system")
            curr_message = SystemMessage(content=content)
            
        elif role == 'user':
            # print("hi user")
            curr_message=UserMessage(content=content)
            
        elif role == 'assistant':
            # Handle function calls in assistant messages
            # print("hi 0")
            if entry.get('function_call'):
                function_call = entry.get('function_call')
                tool_calls = [
                    ToolCall(
                        id=re.sub(r'[^a-zA-Z0-9]', '', function_call.get('tool_call_id')[5:])[:9],
                        function=FunctionCall(
                            name=function_call.get('name'),
                            arguments=function_call.get('arguments')
                        )
                    )
                ]
                curr_message=AssistantMessage(content=None, tool_calls=tool_calls)
                # print("hi 1")
            else:
                curr_message=AssistantMessage(content=content)
                # print("hi 2")
                
        elif role == 'function':
            # print("hi tool_message")
            # Handle function responses as tool messages
            curr_message = ToolMessage(
                    content=content,
                    tool_call_id=re.sub(r'[^a-zA-Z0-9]', '', function_call.get('tool_call_id'))[:9],
                    name=entry.get('name')
                )
        pprint(curr_message)
        print("="*50)
        mistral_messages.append(curr_message)
    return mistral_messages

def process_dataset(input_file: str, output_file: str = None):
    """Process entire dataset and tokenize with Mistral's tokenizer."""
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    dataset = load_jsonl(input_file)
    print(f"Loaded {len(dataset)} conversations")
    
    # Initialize Mistral tokenizer
    tokenizer = MistralTokenizer.v3()
    
    # Process each conversation
    all_tokenized = []
    for i, conversation_data in enumerate(dataset):
        if isinstance(conversation_data, list):
            conversation = conversation_data
        else:
            # If each JSONL entry contains a conversation field
            conversation = conversation_data.get('messages', [])
        
        # Extract tools
        # tools = extract_tools_from_conversation(conversation)
        tools = get_tool_definitions()
        
        # pprint(conversation)
        # Convert to Mistral message objects
        mistral_messages = convert_to_mistral_messages(conversation)
        # pprint(mistral_messages)
        
        if mistral_messages:
            try:
                # Create a ChatCompletionRequest object
                request = ChatCompletionRequest(
                    model="mistral-large-latest",
                    messages=mistral_messages
                    # tools=tools
                )
                # pprint(request.messages)
                # Tokenize using Mistral's tokenizer
                tokenized = tokenizer.encode_chat_completion(request)
                tokens, text = tokenized.tokens, tokenized.text
                
                print(text)

                all_tokenized.append({
                    'conversation_id': i,
                    'tokens': tokens,
                    'text': text,
                    'message_count': len(mistral_messages),
                    'token_count': len(tokens)
                })
                
                if i % 100 == 0:
                    print(f"Processed {i} conversations")
            except Exception as e:
                print(f"Error tokenizing conversation {i}: {e}")
    
    print(f"Successfully tokenized {len(all_tokenized)} conversations")
    
    # Save tokenized data if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_tokenized:
                item_to_save = {
                    'conversation_id': item['conversation_id'],
                    'tokens': item['tokens'],
                    'token_count': item['token_count'],
                    'message_count': item['message_count']
                    # Skip saving text as it might be very large
                }
                f.write(json.dumps(item_to_save) + '\n')
        print(f"Saved tokenized data to {output_file}")
    
    return all_tokenized

if __name__ == "__main__":
    input_file = "chunk.jsonl"
    output_file = "mistral_tokenized_logs.jsonl"
    
    tokenized_data = process_dataset(input_file, output_file)
    
    # Print sample of first tokenized conversation
    if tokenized_data:
        print("\nSample of first tokenized conversation:")
        print(f"Conversation ID: {tokenized_data[0]['conversation_id']}")
        print(f"Number of tokens: {tokenized_data[0]['token_count']}")
        print(f"First 10 tokens: {tokenized_data[0]['tokens'][:10]}")
