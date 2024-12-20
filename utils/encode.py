import tiktoken
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing import List


def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    """
    Get a tokenizer instance by name.

    :param tokenizer_name: tokenizer name, tiktoken encoding name or Hugging Face model name
    :return: tokenizer instance
    """
    if tokenizer_name in tiktoken.list_encoding_names():
        return tiktoken.get_encoding(tokenizer_name)
    elif TRANSFORMERS_AVAILABLE:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from Hugging Face: {e}")
    else:
        raise ValueError("Hugging Face Transformers is not available, please install it first.")

model_name = "cl100k_base"
tokenizer = get_tokenizer(model_name)

def encode_string(text: str) -> List[str]:
    """
    Encode text to tokens

    :param text: text
    :return: tokens
    """
    return tokenizer.encode(text)

def decode_tokens(tokens: List[str]) -> str:
    """
    Decode tokens to text

    :param tokens: tokens
    :return: text
    """
    return tokenizer.decode(tokens)
