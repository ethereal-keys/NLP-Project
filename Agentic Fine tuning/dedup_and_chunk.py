import json
import logging
import tiktoken
import random
import re
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename="chunking_process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # Adjust if using a different tokenizer

def count_tokens(text):
    """Count number of tokens in a text string."""
    if not text:
        return 0
    if isinstance(text, dict) or isinstance(text, list):
        text = json.dumps(text)
    return len(tokenizer.encode(text))

def calculate_conversation_tokens(conversation):
    """Calculate the total token count of a conversation."""
    total_tokens = 0
    for message in conversation:
        if isinstance(message, dict):
            if "content" in message and message["content"]:
                total_tokens += count_tokens(message["content"])
            if "function_call" in message:
                total_tokens += count_tokens(message["function_call"])
    return total_tokens

def clean_system_message(message):
    """
    Clean system message by:
    1. Removing the 'functions available' section from content
    2. Removing the 'tools' key entirely
    
    Returns a copy of the message with cleaned content.
    """
    if message.get("role") != "system":
        return message
    
    # Create a copy of the message to avoid modifying the original
    cleaned_message = message.copy()
    
    # Remove "tools" key if it exists
    if "tools" in cleaned_message:
        del cleaned_message["tools"]
    
    # Clean content if it exists
    if "content" in cleaned_message and cleaned_message["content"]:
        content = cleaned_message["content"]
        # Remove "The following functions are available" section and everything after it
        pattern = r"The following functions are available for you to call.*"
        cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL)
        # Trim any trailing whitespace
        cleaned_content = cleaned_content.rstrip()
        cleaned_message["content"] = cleaned_content
    
    return cleaned_message

def deduplicate_user_messages(messages):
    """
    Remove duplicate user messages entirely while preserving conversation structure.
    If a user message's content is identical to a previous user message, remove the entire message.
    """
    seen_user_contents = set()
    deduplicated_messages = []
    
    for msg in messages:
        if msg.get("role") == "user" and "content" in msg and isinstance(msg["content"], str):
            # If we've seen this content before, skip this message entirely
            if msg["content"] in seen_user_contents:
                continue
            else:
                # New content, add to seen set and keep the message
                seen_user_contents.add(msg["content"])
                deduplicated_messages.append(msg)
        else:
            # Non-user messages or messages without content strings are kept as is
            deduplicated_messages.append(msg)
    
    return deduplicated_messages

def chunk_conversation(messages, max_tokens=4096, min_chunk_size=3):
    """
    Chunk a conversation into smaller parts that fit within token limits.
    Each chunk will start with a cleaned system message, maintain logical tool call sequences,
    and remove duplicate user messages.
    """
    if not messages:
        return []
    
    # Extract and clean system message
    system_message = None
    for msg in messages:
        if msg.get("role") == "system":
            system_message = clean_system_message(msg)
            break
    
    if not system_message:
        # Create a default system message if none exists
        system_message = {
            "role": "system",
            "content": "You are a coding assistant with access to tools."
        }
    
    chunks = []
    current_chunk = [system_message]
    current_token_count = count_tokens(system_message)
    
    i = 0
    while i < len(messages):
        # Skip the system message as we've already added it
        if messages[i].get("role") == "system":
            i += 1
            continue
        
        # Process conversation in logical sequences:
        # We add messages until we reach token limit or end of conversation
        sequence = []
        sequence_tokens = 0
        
        # Extract a logical sequence (user -> assistant -> function)
        while i < len(messages):
            msg = messages[i]
            msg_tokens = count_tokens(msg)
            role = msg.get("role", "")
            
            # Add message to current sequence
            sequence.append(msg)
            sequence_tokens += msg_tokens
            i += 1
            
            # Check if we've reached a natural break point (end of function response)
            # or the end of conversation
            if (role == "function" or i >= len(messages) or 
                (role == "assistant" and "function_call" not in msg)):
                break
        
        # Check if adding this sequence would exceed token limit
        if current_token_count + sequence_tokens > max_tokens and len(current_chunk) > min_chunk_size:
            # Deduplicate before finalizing the chunk
            deduplicated_chunk = deduplicate_user_messages(current_chunk)
            
            # Recalculate token count after deduplication
            deduplicated_token_count = calculate_conversation_tokens(deduplicated_chunk)
            
            # Add if we have a valid chunk
            if len(deduplicated_chunk) > min_chunk_size:
                chunks.append(deduplicated_chunk)
            
            # Start a new chunk with the system message and current sequence
            current_chunk = [system_message] + sequence
            current_token_count = count_tokens(system_message) + sequence_tokens
        else:
            # Add sequence to current chunk
            current_chunk.extend(sequence)
            current_token_count += sequence_tokens
    
    # Add the last chunk if it's not empty and has more than just the system message
    if len(current_chunk) > 1:
        # Deduplicate the last chunk too
        deduplicated_last_chunk = deduplicate_user_messages(current_chunk)
        if len(deduplicated_last_chunk) > min_chunk_size:
            chunks.append(deduplicated_last_chunk)
    
    return chunks

def process_dataset(input_path, output_path, max_tokens=4096, max_examples=None):
    """Process the dataset to create smaller, token-limited examples with deduplication."""
    total_examples = 0
    total_chunks = 0
    token_distribution = []
    tools_removed_count = 0
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Read all lines first to show progress
        lines = infile.readlines()
        lines_to_process = lines[:max_examples] if max_examples else lines
        
        for line in tqdm(lines_to_process, desc="Processing conversations"):
            try:
                example = json.loads(line.strip())
                messages = example.get("messages", [])
                total_examples += 1
                
                if not messages:
                    logger.warning(f"Empty messages in example {total_examples}")
                    continue
                
                # Track if tools were removed
                for msg in messages:
                    if msg.get("role") == "system" and "tools" in msg:
                        tools_removed_count += 1
                        break
                
                # Split into chunks
                chunks = chunk_conversation(messages, max_tokens=max_tokens)
                for chunk in chunks:
                    # Make sure there are at least a few messages in the chunk
                    if len(chunk) < 3:
                        continue
                    
                    # Calculate token count for statistics
                    chunk_tokens = calculate_conversation_tokens(chunk)
                    token_distribution.append(chunk_tokens)
                    
                    # Write chunk to output
                    outfile.write(json.dumps({"messages": chunk}) + "\n")
                    total_chunks += 1
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON error in example {total_examples}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in example {total_examples}: {e}")
    
    # Print statistics
    print(f"Processed {total_examples} examples into {total_chunks} chunks")
    print(f"Removed 'tools' key from {tools_removed_count} system messages")
    if token_distribution:
        print(f"Token statistics:")
        print(f"  Min: {min(token_distribution)}")
        print(f"  Max: {max(token_distribution)}")
        print(f"  Average: {sum(token_distribution) / len(token_distribution):.2f}")
        print(f"  Exceeding {max_tokens} tokens: {sum(1 for t in token_distribution if t > max_tokens)}")
    
    return total_chunks

def validate_dataset(input_path, sample_size=10):
    """Validate a sample of the chunked dataset to verify token counts and check for processing."""
    token_counts = []
    duplicate_stats = {"total": 0, "with_duplicates": 0}
    system_message_stats = {"total": 0, "with_function_docs": 0, "with_tools_key": 0}
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
        # Take a random sample if there are more than sample_size examples
        if len(lines) > sample_size:
            sample_lines = random.sample(lines, sample_size)
        else:
            sample_lines = lines
        
        for i, line in enumerate(sample_lines):
            try:
                example = json.loads(line.strip())
                messages = example.get("messages", [])
                
                # Check if there are any duplicate user contents (this shouldn't happen after processing)
                user_contents = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
                has_duplicates = len(user_contents) != len(set([c for c in user_contents if c]))
                
                if has_duplicates:
                    duplicate_stats["with_duplicates"] += 1
                duplicate_stats["total"] += 1
                
                # Check system messages for function documentation and tools key
                for msg in messages:
                    if msg.get("role") == "system":
                        system_message_stats["total"] += 1
                        if "The following functions are available" in msg.get("content", ""):
                            system_message_stats["with_function_docs"] += 1
                        if "tools" in msg:
                            system_message_stats["with_tools_key"] += 1
                
                # Calculate token count
                token_count = calculate_conversation_tokens(messages)
                token_counts.append(token_count)
                
                # Print detailed information for this sample
                print(f"Sample {i+1}: {token_count} tokens, {len(messages)} messages")
                print(f"  - Duplicates: {'Yes' if has_duplicates else 'No'}")
                
                # Check if system message has been processed correctly
                sys_msg = next((m for m in messages if m.get("role") == "system"), None)
                if sys_msg:
                    print(f"  - System message has 'tools' key: {'Yes' if 'tools' in sys_msg else 'No'}")
                    has_func_text = "The following functions are available" in sys_msg.get("content", "")
                    print(f"  - System message has function text: {'Yes' if has_func_text else 'No'}")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
    
    if token_counts:
        print(f"\nValidation complete: Average token count {sum(token_counts) / len(token_counts):.2f}")
        print(f"Samples with duplicate user messages: {duplicate_stats['with_duplicates']}/{duplicate_stats['total']}")
        print(f"System messages with function documentation: {system_message_stats['with_function_docs']}/{system_message_stats['total']}")
        print(f"System messages with 'tools' key: {system_message_stats['with_tools_key']}/{system_message_stats['total']}")
    else:
        print("No valid samples found")

if __name__ == "__main__":
    input_path = "simplified_swe_agent_logs.jsonl"  # Your cleaned dataset
    output_path = "func_chunked_swe_agent_logs.jsonl"  # Output with smaller chunks
    
    # Process the dataset with 4096 token limit per chunk (as requested)
    chunks = process_dataset(input_path, output_path, max_tokens=4096)
    
    # Change role=function to role=user in the processed dataset
    with open(output_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    modified_lines = []
    for line in lines:
        example = json.loads(line.strip())
        messages = example.get("messages", [])
        
        # Convert all function roles to user roles
        # for message in messages:
        #     if message.get("role") == "function":
        #         message["role"] = "user"
        
        modified_lines.append(json.dumps(example) + "\n")
    
    # Write the modified data back to the file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)
    
    # Validate a sample of the output
    validate_dataset(output_path, sample_size=20)  # Increased sample size for better validation