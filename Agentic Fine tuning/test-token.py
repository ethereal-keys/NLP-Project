from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint
import inspect

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "./chunk.jsonl"
# DATASET_PATH = "./chunked_swe_agent_logs.jsonl"
dataset = load_dataset("json", data_files=DATASET_PATH)
dataset = dataset["train"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function_for_tool_calling(messages):
        
    
    # pprint(messages['messages'], width=2000)
    convo = messages["messages"]

    for i in range(len(convo)):
        convo[i]["content"] = "### SOME CONTENT ###"
        # if convo[i]["role"] == "system"

    pprint(convo)

    # tool = convo[0]["tools"] if convo[0]["tools"] else None
    text = tokenizer.apply_chat_template(
        messages["messages"],
        # tools=tool,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}



dataset = dataset.map(preprocess_function_for_tool_calling, remove_columns=["messages"], num_proc=1)
pprint(dataset["text"], width=120)