import os
import torch
import psutil
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from accelerate import Accelerator
import logging
from pathlib import Path
from transformers import Trainer
import random
import numpy as np
import shutil
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

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "/home/srkashy1/SWE-agent/func_chunked_swe_agent_logs.jsonl"  # Matches dataset script
OUTPUT_DIR = "./codellama-7b"
MAX_SEQ_LENGTH = 4096
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 3 
LEARNING_RATE = 1e-4
WARMUP_STEPS = 200
LOGGING_STEPS = 10
SAVE_STEPS = 100  # Frequent saves
MAX_STEPS = 2340  
DATALOADER_NUM_WORKERS = 8
SAVE_TOTAL_LIMIT = 3  # Keep last 3 checkpoints

# Create output directory first
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

# Set up logging after creating OUTPUT_DIR
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

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


# Check for existing checkpoints to determine resume point
def get_last_checkpoint_step():
    """Find the last checkpoint and extract its step number."""
    if not os.path.exists(OUTPUT_DIR):
        return None
    
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # Extract step numbers and find the maximum
    steps = [int(d.split("-")[1]) for d in checkpoints]
    return max(steps) if steps else None

last_step = get_last_checkpoint_step()
resume_from_checkpoint = os.path.join(OUTPUT_DIR, f"checkpoint-{last_step}") if last_step else None

if resume_from_checkpoint:
    logging.info(f"Will resume training from checkpoint at step {last_step}")
else:
    logging.info("Starting training from scratch")

# Initialize Accelerator with proper configuration
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

# Memory Debugging
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    msg = f"CPU Memory: {mem_info.rss / 1024**3:.2f} GB"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            msg += f"\nGPU {i} Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
            msg += f"\nGPU {i} Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
    logging.info(msg)

# Helper function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Load Model with error checking
def load_model_safely():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )
        logging.info("Model loaded successfully with quantization")
        return model
    except RuntimeError as e:
        logging.error(f"Quantization failed: {e}. Falling back to BF16 without quantization.")
        clear_gpu_memory()  # Clear GPU memory before retry
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16
            )
            logging.info("Model loaded with BF16 without quantization")
            return model
        except Exception as e2:
            logging.critical(f"Failed to load model even without quantization: {e2}")
            raise RuntimeError("Model loading failed completely")

# Load model
model = load_model_safely()
print_memory_usage()

# Load Tokenizer and Add Special Tokens (including tool calling tokens)
try:
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.padding_side = "right"  # Important for proper padding
    
    # # Explicitly set pad_token and eos_token to avoid inference issues
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is not None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #         logging.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    #     else:
    #         tokenizer.pad_token = "<|endoftext|>"
    #         tokenizer.eos_token = "<|endoftext|>"
    #         logging.info("Set pad_token and eos_token to '<|endoftext|>'")
    tokenizer = MistralTokenizer.v3()
    
except Exception as e:
    logging.critical(f"Failed to load or configure tokenizer: {e}")
    raise

# LoRA Config - fixed target modules based on model architecture
# Check model architecture to determine proper target modules
model_modules = set()
for name, _ in model.named_modules():
    model_modules.add(name)

# Determine appropriate target modules based on model architecture
if any("q_proj" in module for module in model_modules):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    logging.info("Using standard attention target modules for LoRA")
else:
    # Fallback for different architectures - check first few thousand modules to find patterns
    module_samples = list(model_modules)[:1000]
    logging.info(f"Model doesn't have standard attention modules, checking alternatives...")
    # Example fallback - you might need to adjust based on specific model architecture
    target_modules = ["query", "key", "value"] if any("query" in m for m in module_samples) else ["query_key_value"]
    logging.info(f"Using alternative target modules for LoRA: {target_modules}")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
try:
    model = get_peft_model(model, lora_config)
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(f"Trainable params: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of {all_params:,d})")
    
    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    logging.info("Applied LoRA successfully")
    print_memory_usage()
except Exception as e:
    logging.critical(f"Failed to apply LoRA: {e}")
    raise

# Load Dataset with validation
try:
    dataset = load_dataset("json", data_files=DATASET_PATH)
    dataset = dataset["train"]
    
    # Validate dataset - check if it has the expected format
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Check for 'messages' column
    if "messages" not in dataset.column_names:
        raise ValueError(f"Dataset missing 'messages' column. Available columns: {dataset.column_names}")
    
    # Check sample messages
    sample = dataset[0]["messages"]
    if not isinstance(sample, list):
        raise ValueError(f"Expected 'messages' to be a list, got {type(sample)}")
    
    logging.info(f"Loaded dataset with {len(dataset)} training examples")
except Exception as e:
    logging.critical(f"Failed to load dataset: {e}")
    raise

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
                        id=re.sub(r'[^a-zA-Z0-9]', '', function_call.get('tool_call_id'))[:9],
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


# Preprocess Dataset with better error handling
def preprocess_function_for_tool_calling(example):
    """Process entire dataset and tokenize with Mistral's tokenizer."""
    
    # Initialize Mistral tokenizer
    tokenizer = MistralTokenizer.v3()
    
    # Process each conversation
    conversation=example["messages"]
    # Extract tools
    # tools = get_tool_definitions()
    
    # pprint(conversation)
    # Convert to Mistral message objects
    mistral_messages = convert_to_mistral_messages(conversation)
    # pprint(mistral_messages)
    
    if mistral_messages:
        try:
            # Create a ChatCompletionRequest object
            request = ChatCompletionRequest(
                model=MODEL_NAME,
                messages=mistral_messages
                # tools=tools
            )
            # print(request)
            # Tokenize using Mistral's tokenizer
            # tokenized = tokenizer.encode_chat_completion(request)
            # tokens, text = tokenized.tokens, tokenized.text
            
            # all_tokenized.append({
            #     'conversation_id': i,
            #     'tokens': tokens,
            #     'text': text,
            #     'message_count': len(mistral_messages),
            #     'token_count': len(tokens)
            # })
            
            # if i % 100 == 0:
            #     print(f"Processed {i} conversations")
        except Exception as e:
            print(f"Error tokenizing conversation {i}: {e}")
    
    return {"text": request}
        ##############################################################################

# Sample format validation for tool calling datasets
def validate_tool_calling_dataset(dataset):
    """Validate that the dataset contains properly formatted tool calling examples."""
    has_tool_calls = False
    has_tool_definitions = False
    
    # Check a sample of examples
    sample_size = min(100, len(dataset))
    for i in range(sample_size):
        example = dataset[i]
        if "messages" not in example:
            continue
            
        messages = example["messages"]
        
        # Check for tool definitions in system messages
        for msg in messages:
            if msg.get("role") == "system" and "tools" in msg:
                has_tool_definitions = True
                
            # Check for function calls in assistant messages
            if msg.get("role") == "assistant" and "function_call" in msg:
                has_tool_calls = True
                
            # Check for function responses
            if msg.get("role") == "function":
                has_tool_calls = True
                
    if not has_tool_definitions:
        logging.warning("Dataset may not contain tool definitions in system messages")
    
    if not has_tool_calls:
        logging.warning("Dataset may not contain function calls or function responses")
        
    if has_tool_definitions and has_tool_calls:
        logging.info("Dataset contains tool definitions and function calls - good for tool calling training")
    
    return has_tool_definitions and has_tool_calls

try:
    # First validate dataset for tool calling
    # is_valid_tool_dataset = validate_tool_calling_dataset(dataset)
    # if not is_valid_tool_dataset:
    #     logging.warning("Dataset may not be properly formatted for tool calling - check your data")
    
    # Process with the tool-calling-aware preprocessing function
    dataset = dataset.map(preprocess_function_for_tool_calling, remove_columns=["messages"], num_proc=4)
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x["text"] is not None)
    filtered_size = len(dataset)
    
    if filtered_size < original_size:
        logging.warning(f"Filtered out {original_size - filtered_size} examples due to preprocessing errors")
    
    if filtered_size == 0:
        raise ValueError("All examples were filtered out! Check your dataset format.")
    
    logging.info(f"Preprocessed dataset with {filtered_size} valid examples for tool calling")
    
    # Add this to examine a processed example
    if filtered_size > 0:
        sample_text = dataset[0]["text"]
        logging.info(f"Sample processed text (truncated to 500 chars):\n{sample_text[:500]}...")
        
except Exception as e:
    logging.critical(f"Failed to preprocess dataset: {e}")
    raise

# Custom helper for checkpoint resumption 
def find_global_step_from_checkpoint(checkpoint_dir):
    """Extract the global step from a checkpoint directory name"""
    try:
        if not checkpoint_dir:
            return 0
        
        if os.path.isdir(checkpoint_dir):
            # Extract step from directory name if in format "checkpoint-X"
            if "checkpoint-" in os.path.basename(checkpoint_dir):
                step = int(os.path.basename(checkpoint_dir).split("-")[1])
                return step
            
            # Try to read from state file
            state_file = os.path.join(checkpoint_dir, "trainer_state.json")
            if os.path.exists(state_file):
                import json
                with open(state_file, "r") as f:
                    state = json.load(f)
                    if "global_step" in state:
                        return state["global_step"]
        
        return 0
    except Exception as e:
        logging.warning(f"Failed to find global step from checkpoint: {e}")
        return 0

# Get the global step if resuming from checkpoint
global_step = find_global_step_from_checkpoint(resume_from_checkpoint)
logging.info(f"Starting or resuming from global step: {global_step}")

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_strategy="steps",
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_on_each_node=True,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    report_to="tensorboard",
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    disable_tqdm=False,
    # Add ddp_find_unused_parameters to avoid hanging
    ddp_find_unused_parameters=False,
)

# Custom callback for checkpoint management
class CheckpointCallback(TrainerCallback):
    def __init__(self, initial_step=0):
        self.initial_step = initial_step
        
    def on_save(self, args, state, control, **kwargs):
        # Adjust the checkpoint directories to maintain consistent numbering
        # when resuming from checkpoint
        if hasattr(state, "global_step"):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            logging.info(f"Checkpoint saved at {checkpoint_dir}")
            
            # Add explicit tracker file to remember true global step (helps with resumption)
            tracker_file = os.path.join(checkpoint_dir, "global_step.txt")
            with open(tracker_file, "w") as f:
                f.write(str(state.global_step))
        
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Log the starting step
        logging.info(f"Training beginning with initial step: {self.initial_step}")
        
        # Update the state with the correct initial step
        if hasattr(state, "global_step") and state.global_step == 0 and self.initial_step > 0:
            state.global_step = self.initial_step
            logging.info(f"Updated state.global_step to {state.global_step}")
        
        return control

original_load_rng_state = Trainer._load_rng_state

# Define a new version that handles the PyTorch 2.6+ behavior
def patched_load_rng_state(self, checkpoint):
    """Patched version of _load_rng_state that handles weights_only parameter"""
    try:
        # Get the RNG file path
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if os.path.isfile(rng_file):
            try:
                # Critical fix: explicitly set weights_only=False
                checkpoint_rng_state = torch.load(rng_file, weights_only=False)
                
                # Set the RNG states
                random.setstate(checkpoint_rng_state["python"])
                np.random.set_state(checkpoint_rng_state["numpy"])
                torch.random.set_rng_state(checkpoint_rng_state["cpu"])
                if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
                    if hasattr(torch.cuda, "set_rng_state_all"):
                        torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])
                    else:
                        for i, state in enumerate(checkpoint_rng_state["cuda"]):
                            if i < torch.cuda.device_count():
                                torch.cuda.set_rng_state(state)
                
                logging.info("RNG state successfully loaded from checkpoint")
            except Exception as e:
                logging.warning(f"Failed to load RNG state with error: {e}")
                logging.warning("Training will continue with a fresh RNG state")
    except Exception as e:
        logging.warning(f"Error while loading RNG state: {e}")
        logging.warning("Training will continue with a fresh RNG state")

# Apply the monkey patch
Trainer._load_rng_state = patched_load_rng_state

# Create Dataset collator fixing attention mask issues
class AttentionAwareDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        # This is a simplified version; SFTTrainer has its own collator
        # but if needed, you can implement a custom one like this
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )
        
        # Explicitly set attention_mask for proper padding
        if "input_ids" in batch and "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            # Set attention mask to 0 for pad tokens
            batch["attention_mask"] = batch["attention_mask"].masked_fill(
                batch["input_ids"] == self.tokenizer.pad_token_id, 0
            )
        
        return batch

# SFT Trainer setup with proper error handling
try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )
    
    # Add custom checkpoint callback with the correct initial step
    trainer.add_callback(CheckpointCallback(initial_step=global_step))
    
    # Manually set the global step if resuming
    if global_step > 0:
        trainer.state.global_step = global_step
        logging.info(f"Set trainer's global_step to {global_step}")
    
    logging.info("Initialized SFTTrainer successfully")
except Exception as e:
    logging.critical(f"Failed to initialize SFTTrainer: {e}")
    clear_gpu_memory()
    raise

# Train with better error handling and cleanup
try:
    # Check for CUDA availability before training
    if torch.cuda.is_available():
        logging.info(f"Training with {torch.cuda.device_count()} GPU(s)")
    else:
        logging.warning("No GPU detected, training on CPU")
    
    # Train with resumption if needed
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully")
except Exception as e:
    logging.critical(f"Training failed: {e}")
    # Proper cleanup
    clear_gpu_memory()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    raise
finally:
    # Always clean up memory regardless of success or failure
    clear_gpu_memory()

# Save Final Model
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
try:
    # Make sure the directory exists
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save model and tokenizer
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save a record of the final step
    with open(os.path.join(final_model_path, "final_step.txt"), "w") as f:
        f.write(str(trainer.state.global_step))
    
    logging.info(f"Final model saved at {final_model_path}")
    
    # Clean up distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
except Exception as e:
    logging.critical(f"Failed to save final model: {e}")
    raise

logging.info("Training pipeline completed successfully!")