import json

from models.llm.tokenizer import Tokenizer

tokenizer = Tokenizer(
    model_name="Qwen/Qwen2.5-72B-Instruct"
)

def get_dataset_stats(dataset_path):
    with open(dataset_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]

    if not "content" in data[0]:
        for item in data:
            item["content"] = item["answer"]

    token_lengths = [len(tokenizer.encode_string(item["content"])) for item in data]

    print(f"Number of samples: {len(token_lengths)}")
    print(f"Average token length: {sum(token_lengths) / len(token_lengths)}")
    print(f"Max token length: {max(token_lengths)}")

if __name__ == "__main__":
    get_dataset_stats("/home/PJLAB/chenzihong/Downloads/1-4.json")