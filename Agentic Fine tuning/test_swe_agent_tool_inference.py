import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import textwrap
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

# Configuration
BASE_MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
LORA_PATH = "./codellama-7b/checkpoint-300"  # Update to your saved LoRA path
MAX_NEW_TOKENS = 1024  # Maximum generation length
TEMPERATURE = 0.2  # Lower for more deterministic outputs

# Sample prompts template - system prompt and user query
# These are placeholders, you can modify these later with actual content
SYSTEM_PROMPT = """
You are a helpful assistant.
"""

USER_PROMPT = """
Help me create a function that sorts a list of dictionaries by a specific key.
"""

# Function to load the fine-tuned model
def load_finetuned_model():
    """Load the base model with the LoRA adapter"""
    print("Loading tokenizer...")
    # First load the tokenizer from the LoRA path to get any special tokens
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Important: Resize the model's embeddings BEFORE loading the LoRA adapter
    # print(f"Resizing token embeddings from {base_model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
    # base_model.resize_token_embeddings(len(tokenizer))
    
    # print("Loading LoRA adapter...")
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     LORA_PATH,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )
    
    # # Explicitly set pad token if it's not defined
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # print("Model loaded successfully!")
    return model, tokenizer

# Function to format the prompt for the model
def format_prompt(system_prompt, user_prompt):
    """Format the prompt according to model's chat template"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

# Sample tool definitions to include in system prompt
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

# Add tool definitions to system message
def add_tools_to_system_message(system_prompt, tools):
    """Add tools to the system message"""
    system_message = {"role": "system", "content": system_prompt, "tools": tools}
    return system_message

# Format the assistant's response to be readable
def format_response(response_text):
    """Format the response for better readability"""
    # Implement any specific formatting needed
    return response_text

# Parse function calls from the response
def parse_function_calls(response_text):
    """Parse and extract function calls from the model's response"""
    # This is a placeholder - actual parsing depends on the model's output format
    # For Qwen models, you'd need to check their specific function calling format
    
    # Example extraction logic (adjust based on actual model output):
    function_calls = []
    
    # Basic detection for function call patterns - this is very simplistic
    # and should be adapted to your model's actual format
    if "<|function_call|>" in response_text:
        try:
            # Split by function call markers
            parts = response_text.split("<|function_call|>")
            for part in parts[1:]:  # Skip the first part (before any function call)
                if "|>" in part:
                    func_text = part.split("|>")[0].strip()
                    # Try to parse as JSON
                    try:
                        func_data = json.loads(func_text)
                        function_calls.append(func_data)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error parsing function calls: {e}")
    
    return function_calls

# Test the model with a prompt
def test_model(model, tokenizer, system_prompt, user_prompt):
    """Test the model with a given prompt"""
    # Add tool definitions to system message
    tools = get_tool_definitions()
    system_message = add_tools_to_system_message(system_prompt, tools)
    
    # Create the full messages list
    messages = [
        system_message,
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n" + "="*50)
    print("PROMPT:")
    print("="*50)
    print(textwrap.fill(prompt, width=100))
    
    # Tokenize and generate
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    print(input_ids.shape)
    attention_mask = inputs.attention_mask.to(model.device)

    # Set generation parameters
    gen_params = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,  # Add attention mask
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": TEMPERATURE > 0,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,  # Explicitly set pad token ID
        "eos_token_id": tokenizer.eos_token_id,  # Explicitly set EOS token ID
    }
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**gen_params)

    # print(outputs)
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)

    # Clean up any remaining FIM tokens that might not be caught by skip_special_tokens
    fim_tokens = ["<|fim_pad|>", "<|fim*pad|>", "<|im_start|>", "<|im_end|>"]
    for token in fim_tokens:
        response = response.replace(token, "")

    # Extract the assistant's message from the full response
    if "<|im_start|>assistant" in response:
        assistant_response = response.split("<|im_start|>assistant")[1].strip()
    else:
        # Extract based on the applied chat template
        assistant_response = response[len(prompt):].strip()
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(textwrap.fill(assistant_response, width=100))
    
    # Parse function calls if present
    function_calls = parse_function_calls(assistant_response)
    if function_calls:
        print("\n" + "="*50)
        print("DETECTED FUNCTION CALLS:")
        print("="*50)
        for i, func in enumerate(function_calls):
            print(f"Function {i+1}:")
            print(json.dumps(func, indent=2))
    
    return assistant_response, function_calls

# Main function to run the demo
def main():
    # Load the model
    model, tokenizer = load_finetuned_model()
    
    # Test with sample prompt
    test_model(model, tokenizer, SYSTEM_PROMPT, USER_PROMPT)
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        print("Enter a prompt (or 'q' to quit):")
        user_input = input("> ")
        if user_input.lower() in ('q', 'quit', 'exit'):
            break
        
        test_model(model, tokenizer, SYSTEM_PROMPT, user_input)

if __name__ == "__main__":
    main()
