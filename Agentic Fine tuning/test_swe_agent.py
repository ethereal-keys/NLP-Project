import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import glob

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
CHECKPOINT_DIR = "./qwen2.5-coder-0.5b-sft-swe"
SPECIAL_TOKENS = ["<|INPUT|>", "<|OUTPUT|>", "<|END|>"]

# 4-bit Quantization Config (matching training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def find_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint folder with the highest step number."""
    checkpoint_folders = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    print(f"Available checkpoints: {checkpoint_folders}")
    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract step numbers and find the highest
    checkpoint_steps = [
        int(folder.split("checkpoint-")[-1]) for folder in checkpoint_folders
        if folder.split("checkpoint-")[-1].isdigit()
    ]
    if not checkpoint_steps:
        raise ValueError("No valid checkpoint folders found (e.g., checkpoint-200)")
    
    latest_step = max(checkpoint_steps)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint-{latest_step}")
    return latest_checkpoint

def load_lora_model():
    """Load the base model and apply LoRA adapters from the latest checkpoint."""
    # Find latest checkpoint
    try:
        latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
        print(f"Loading LoRA adapters from: {latest_checkpoint}")
    except Exception as e:
        print(f"Error finding checkpoint: {e}")
        return None, None

    # Load base model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Base model loaded successfully")
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return None, None

    # Load tokenizer and add special tokens
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "right"
        new_tokens = [token for token in SPECIAL_TOKENS if token not in tokenizer.get_vocab()]
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added} special tokens to tokenizer")
            model.resize_token_embeddings(len(tokenizer))
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return None, None

    # Load LoRA adapters
    try:
        model = PeftModel.from_pretrained(model, latest_checkpoint)
        print("LoRA adapters applied successfully")
    except Exception as e:
        print(f"Failed to load LoRA adapters: {e}")
        return None, None

    return model, tokenizer

def test_inference(model, tokenizer):
    """Run a test inference with a tool-calling prompt."""
    if model is None or tokenizer is None:
        print("Cannot run inference: model or tokenizer not loaded")
        return

    # Test prompt (similar to training dataset)
    prompt = """<|INPUT|>SYSTEM: You are an autonomous programmer in a SWE-Agent environment. You can execute shell commands and programming tasks to solve issues efficiently and correctly. Use the following format for tool-calling actions:
- [CREATE file_path]: Create a new file with the specified content.
- [EDIT file_path]: Edit an existing file with the specified changes.
- [SUBMIT commit_message]: Submit changes to the repository with the given commit message.
- [RUN command]: Execute a shell command (e.g., run tests).
Provide clear steps, shell commands, and code snippets where applicable.

USER: Create a Python file `utils.py` with a function `calculate_metrics` that computes accuracy and F1 score for a given set of predictions and labels. Then, submit the changes.

Please provide the steps and commands to resolve this issue.<|OUTPUT|>"""

    # Tokenize and generate
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("Model Response:")
        print(response.split("<|OUTPUT|>")[1].split("<|END|>")[0].strip())
    except Exception as e:
        print(f"Inference failed: {e}")

def main():
    """Load and test the LoRA model from the latest checkpoint."""
    model, tokenizer = load_lora_model()
    test_inference(model, tokenizer)

if __name__ == "__main__":
    main()