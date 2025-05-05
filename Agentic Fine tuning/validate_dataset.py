import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import random
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(
    filename="dataset_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

def analyze_dataset(dataset_path, model_name="Qwen/Qwen2.5-Coder-0.5B", max_token_length=2048, 
                   sample_size=100, detailed_examples=3, error_examples=3):
    """
    Thoroughly analyze a tool calling dataset for quality and correctness.
    
    Args:
        dataset_path: Path to the JSONL dataset
        model_name: Model name for tokenizer
        max_token_length: Maximum allowed token length
        sample_size: Number of examples to analyze in detail
        detailed_examples: Number of examples to print in detail
        error_examples: Number of error examples to print
    """
    logger.info(f"Starting validation of dataset: {dataset_path}")
    
    # Load tokenizer for token counting
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer for {model_name}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
    
    # Dataset statistics
    examples = []
    line_count = 0
    error_count = 0
    errors = []
    
    # Read all examples
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Reading dataset")):
                line_count += 1
                try:
                    example = json.loads(line.strip())
                    examples.append((line_idx, example))
                except json.JSONDecodeError:
                    error_count += 1
                    errors.append((line_idx, "JSON decode error", line[:100]))
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return
    
    logger.info(f"Read {line_count} lines with {error_count} JSON errors")
    
    # Basic dataset statistics
    valid_examples = len(examples)
    logger.info(f"Valid examples: {valid_examples}")
    
    if valid_examples == 0:
        logger.error("No valid examples found, stopping validation")
        return
    
    # Select random examples for detailed analysis
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(valid_examples), min(sample_size, valid_examples))
    sample_examples = [examples[i] for i in sample_indices]
    
    # Basic statistics tracking
    stats = {
        "message_counts": [],
        "token_lengths": [],
        "tool_calls_per_example": [],
        "tool_names": Counter(),
        "role_sequences": Counter(),
        "exceeds_token_limit": 0,
        "missing_key_fields": 0,
        "duplicate_tool_call_ids": 0,
        "tool_call_ids_without_responses": 0,
        "function_responses_without_calls": 0,
        "unterminated_conversations": 0,
        "complete_tool_call_sequence": 0,
        "tool_calls_without_tool_schemas": 0,
        "tool_schema_counts": Counter(),
        "broken_conversation_patterns": 0,
    }
    
    conversation_patterns = {
        "assistant_without_user": 0,
        "function_without_assistant": 0,
        "consecutive_users": 0,
        "consecutive_assistants": 0,
        "consecutive_functions": 0,
    }
    
    # Example errors for reporting
    example_errors = defaultdict(list)
    
    # Process each example
    for idx, (line_idx, example) in enumerate(tqdm(examples, desc="Analyzing examples")):
        if "messages" not in example:
            stats["missing_key_fields"] += 1
            example_errors["missing_messages"].append((line_idx, example))
            continue
            
        messages = example["messages"]
        if not isinstance(messages, list):
            stats["missing_key_fields"] += 1
            example_errors["invalid_messages"].append((line_idx, type(messages)))
            continue
        
        stats["message_counts"].append(len(messages))
        
        # Check token length
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer.encode(text)
            token_length = len(tokens)
            stats["token_lengths"].append(token_length)
            
            if token_length > max_token_length:
                stats["exceeds_token_limit"] += 1
                if len(example_errors["token_exceeded"]) < error_examples:
                    example_errors["token_exceeded"].append((line_idx, token_length))
        except Exception as e:
            logger.warning(f"Error encoding example {line_idx}: {e}")
            if len(example_errors["encoding_error"]) < error_examples:
                example_errors["encoding_error"].append((line_idx, str(e)))
        
        # Analyze message sequence
        has_system = False
        has_tool_schema = False
        tool_call_ids = set()
        responded_tool_call_ids = set()
        tool_calls_count = 0
        role_sequence = []
        
        # Check for tool schemas in system message
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                if len(example_errors["invalid_message"]) < error_examples:
                    example_errors["invalid_message"].append((line_idx, i, type(msg)))
                continue
                
            role = msg.get("role", "unknown")
            role_sequence.append(role)
            
            # System message checks
            if role == "system":
                has_system = True
                if "tools" in msg and isinstance(msg["tools"], list):
                    has_tool_schema = True
                    stats["tool_schema_counts"].update([len(msg["tools"])])
            
            # Assistant message checks
            elif role == "assistant":
                if "function_call" in msg and isinstance(msg["function_call"], list):
                    func_calls = msg["function_call"]
                    tool_calls_count += len(func_calls)
                    
                    for call in func_calls:
                        if "tool_call_id" in call:
                            tool_call_id = call["tool_call_id"]
                            if tool_call_id in tool_call_ids:
                                stats["duplicate_tool_call_ids"] += 1
                            tool_call_ids.add(tool_call_id)
                        
                        if "name" in call:
                            stats["tool_names"].update([call["name"]])
            
            # Function message checks
            elif role == "function":
                if "tool_call_id" in msg:
                    tool_call_id = msg["tool_call_id"]
                    responded_tool_call_ids.add(tool_call_id)
                    
                    if tool_call_id not in tool_call_ids:
                        stats["function_responses_without_calls"] += 1
                        if len(example_errors["response_without_call"]) < error_examples:
                            example_errors["response_without_call"].append((line_idx, tool_call_id))
        
        # Update tool call stats
        stats["tool_calls_per_example"].append(tool_calls_count)
        
        # Check if all tool calls have responses
        missing_responses = tool_call_ids - responded_tool_call_ids
        if missing_responses:
            stats["tool_call_ids_without_responses"] += len(missing_responses)
            if len(example_errors["missing_responses"]) < error_examples:
                example_errors["missing_responses"].append((line_idx, list(missing_responses)))
        
        # Check for conversation flow issues
        if not has_system:
            if len(example_errors["no_system"]) < error_examples:
                example_errors["no_system"].append(line_idx)
        
        if not has_tool_schema and tool_calls_count > 0:
            stats["tool_calls_without_tool_schemas"] += 1
            if len(example_errors["no_tool_schema"]) < error_examples:
                example_errors["no_tool_schema"].append(line_idx)
        
        # Check conversation patterns
        role_str = " ".join(role_sequence)
        stats["role_sequences"].update([role_str[:100]])  # Truncate very long sequences
        
        # Check for conversation pattern issues
        for i in range(len(role_sequence) - 1):
            curr_role = role_sequence[i]
            next_role = role_sequence[i+1]
            
            if curr_role == "user" and next_role == "user":
                conversation_patterns["consecutive_users"] += 1
            elif curr_role == "assistant" and next_role == "assistant":
                conversation_patterns["consecutive_assistants"] += 1
            elif curr_role == "function" and next_role == "function":
                conversation_patterns["consecutive_functions"] += 1
            elif curr_role == "user" and next_role != "assistant":
                conversation_patterns["user_not_followed_by_assistant"] += 1
            elif curr_role == "assistant" and next_role != "function" and next_role != "user":
                if "function_call" in messages[i]:
                    conversation_patterns["assistant_with_function_call_not_followed_by_function"] += 1
        
        # Check if conversation ends with user or assistant
        if role_sequence and role_sequence[-1] not in ["user", "assistant"]:
            stats["unterminated_conversations"] += 1
            
        # Check if example follows complete tool call pattern
        if has_tool_schema and tool_calls_count > 0 and len(missing_responses) == 0:
            stats["complete_tool_call_sequence"] += 1
    
    # Print detailed statistics
    logger.info("\n===== DATASET STATISTICS =====")
    logger.info(f"Total examples: {valid_examples}")
    logger.info(f"Total messages: {sum(stats['message_counts'])}")
    logger.info(f"Average messages per example: {sum(stats['message_counts']) / valid_examples:.2f}")
    logger.info(f"Average token length: {sum(stats['token_lengths']) / len(stats['token_lengths']):.2f}")
    logger.info(f"Examples exceeding token limit ({max_token_length}): {stats['exceeds_token_limit']} ({stats['exceeds_token_limit']/valid_examples*100:.2f}%)")
    logger.info(f"Average tool calls per example: {sum(stats['tool_calls_per_example']) / valid_examples:.2f}")
    logger.info(f"Examples with complete tool call sequences: {stats['complete_tool_call_sequence']} ({stats['complete_tool_call_sequence']/valid_examples*100:.2f}%)")
    
    # Token length distribution
    token_lengths = sorted(stats["token_lengths"])
    if token_lengths:
        percentiles = {
            "min": token_lengths[0],
            "25%": token_lengths[len(token_lengths)//4],
            "50%": token_lengths[len(token_lengths)//2],
            "75%": token_lengths[3*len(token_lengths)//4],
            "90%": token_lengths[9*len(token_lengths)//10],
            "95%": token_lengths[95*len(token_lengths)//100],
            "99%": token_lengths[99*len(token_lengths)//100],
            "max": token_lengths[-1]
        }
        logger.info("\n===== TOKEN LENGTH DISTRIBUTION =====")
        for k, v in percentiles.items():
            logger.info(f"{k}: {v}")
    
    # Report most common tools
    logger.info("\n===== TOP 10 TOOL NAMES =====")
    for tool, count in stats["tool_names"].most_common(10):
        logger.info(f"{tool}: {count}")
    
    # Report most common role sequences
    logger.info("\n===== TOP 5 ROLE SEQUENCES =====")
    for seq, count in stats["role_sequences"].most_common(5):
        logger.info(f"{seq}: {count}")
    
    # Report conversation pattern issues
    logger.info("\n===== CONVERSATION PATTERN ISSUES =====")
    for pattern, count in conversation_patterns.items():
        if count > 0:
            logger.info(f"{pattern}: {count}")
    
    # Report errors
    logger.info("\n===== ERROR SUMMARY =====")
    logger.info(f"Missing key fields: {stats['missing_key_fields']}")
    logger.info(f"Duplicate tool call IDs: {stats['duplicate_tool_call_ids']}")
    logger.info(f"Tool calls without responses: {stats['tool_call_ids_without_responses']}")
    logger.info(f"Function responses without calls: {stats['function_responses_without_calls']}")
    logger.info(f"Unterminated conversations: {stats['unterminated_conversations']}")
    logger.info(f"Tool calls without tool schemas: {stats['tool_calls_without_tool_schemas']}")
    
    # Print example errors
    if example_errors:
        logger.info("\n===== ERROR EXAMPLES =====")
        for error_type, examples in example_errors.items():
            if examples:
                logger.info(f"\n{error_type} ({len(examples)} instances):")
                for i, example_data in enumerate(examples[:error_examples]):
                    logger.info(f"  Example {i+1}: Line {example_data[0]}, {example_data[1:]}")
    
    # Print detailed examples
    if detailed_examples > 0:
        logger.info("\n===== DETAILED EXAMPLES =====")
        for i, (line_idx, example) in enumerate(sample_examples[:detailed_examples]):
            logger.info(f"\nEXAMPLE {i+1} (Line {line_idx}):")
            
            messages = example.get("messages", [])
            for j, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    
                    if role == "system":
                        logger.info(f"  {j}. SYSTEM: {msg.get('content', '')[:100]}...")
                        if "tools" in msg:
                            logger.info(f"     Tools defined: {len(msg['tools'])}")
                    
                    elif role == "user":
                        logger.info(f"  {j}. USER: {msg.get('content', '')[:100]}...")
                    
                    elif role == "assistant":
                        logger.info(f"  {j}. ASSISTANT: {msg.get('content', '')[:100]}...")
                        if "function_call" in msg:
                            for k, call in enumerate(msg["function_call"]):
                                logger.info(f"     Function call {k+1}: {call.get('name', 'unknown')}")
                    
                    elif role == "function":
                        logger.info(f"  {j}. FUNCTION: {msg.get('name', 'unknown')}")
                        logger.info(f"     tool_call_id: {msg.get('tool_call_id', 'missing')}")
                        logger.info(f"     content: {msg.get('content', '')[:100]}...")
            
            # Calculate token length for this example
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                token_length = len(tokenizer.encode(text))
                logger.info(f"  Token length: {token_length}")
            except Exception as e:
                logger.info(f"  Token length calculation error: {e}")
    
    logger.info("\n===== VALIDATION COMPLETE =====")
    
    # Return summary for potential programmatic use
    return {
        "valid_examples": valid_examples,
        "avg_messages": sum(stats["message_counts"]) / valid_examples if valid_examples else 0,
        "avg_tokens": sum(stats["token_lengths"]) / len(stats["token_lengths"]) if stats["token_lengths"] else 0,
        "exceeds_token_limit": stats["exceeds_token_limit"],
        "complete_tool_call_sequence_pct": stats["complete_tool_call_sequence"] / valid_examples * 100 if valid_examples else 0,
        "errors": sum([
            stats["missing_key_fields"],
            stats["duplicate_tool_call_ids"],
            stats["tool_call_ids_without_responses"],
            stats["function_responses_without_calls"],
            stats["unterminated_conversations"],
            stats["tool_calls_without_tool_schemas"]
        ])
    }

def main():
    parser = argparse.ArgumentParser(description="Validate a tool calling dataset")
    parser.add_argument("dataset_path", help="Path to the JSONL dataset file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B", help="Model name for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum token length")
    parser.add_argument("--sample", type=int, default=100, help="Number of examples to analyze in detail")
    parser.add_argument("--details", type=int, default=0, help="Number of examples to print in detail")
    parser.add_argument("--error-examples", type=int, default=3, help="Number of error examples to print")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return
    
    results = analyze_dataset(
        args.dataset_path, 
        model_name=args.model,
        max_token_length=args.max_tokens,
        sample_size=args.sample,
        detailed_examples=args.details,
        error_examples=args.error_examples
    )
    
    # Final verdict
    if results:
        if results["errors"] == 0 and results["exceeds_token_limit"] == 0:
            print("\n✅ DATASET VALIDATION PASSED: No critical issues found.")
        elif results["errors"] < results["valid_examples"] * 0.05 and results["exceeds_token_limit"] < results["valid_examples"] * 0.05:
            print(f"\n⚠️ DATASET VALIDATION WARNING: {results['errors']} errors and {results['exceeds_token_limit']} examples exceed token limit.")
            print("   This is a small percentage (<5%) of examples and may be acceptable.")
        else:
            print(f"\n❌ DATASET VALIDATION FAILED: {results['errors']} errors and {results['exceeds_token_limit']} examples exceed token limit.")
            print("   Please fix these issues before training.")

if __name__ == "__main__":
    main()