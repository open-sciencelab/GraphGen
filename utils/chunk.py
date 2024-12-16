from .encode import encode_string, decode_tokens

def chunk_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024
):
    tokens = encode_string(content)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens(
            tokens[start : start + max_token_size]
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results
