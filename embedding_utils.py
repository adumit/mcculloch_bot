import os
from openai import OpenAI


def get_batched_embeddings(
    texts: list[str],
    model: str,
    max_batch_size: int = 500,
) -> list[list[float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    texts = [text.replace("\n", " ") for text in texts]
    embeddings = []
    for i in range(0, len(texts), max_batch_size):
        batch = texts[i : i + max_batch_size]
        batch_embeddings = client.embeddings.create(input=batch, model=model).data
        embeddings.extend([embedding.embedding for embedding in batch_embeddings])
    return embeddings