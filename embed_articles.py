from dotenv import load_dotenv
load_dotenv()

import semchunk
from pydantic import BaseModel
import os
from tqdm import tqdm
import json

from constants import EMBEDDING_MODEL
from embedding_utils import get_batched_embeddings

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "scraped_documents")
EMBEDDED_DOCUMENTS_FILE = os.path.join(SAVE_DIR, "embedded_documents.json")


class Chunk(BaseModel):
    text: str
    embedding: list[float]


class Document(BaseModel):
    chunks: list[Chunk]
    file_name: str



def get_chunks(file_text: str) -> list[Chunk]:
    chunk_size = 700
    chunker = semchunk.chunkerify('gpt-4', chunk_size)
    chunk_strs = [x for x in chunker(file_text) if len(x.split(" ")) > 50]
    embeddings = get_batched_embeddings(chunk_strs, EMBEDDING_MODEL)
    return [Chunk(text=chunk_str, embedding=embedding) for chunk_str, embedding in zip(chunk_strs, embeddings)]


def embed_documents():
    all_files = os.listdir(SAVE_DIR)

    documents = []

    for file in tqdm(all_files):
        with open(os.path.join(SAVE_DIR, file)) as f:
            file_text = f.read()
            chunks = get_chunks(file_text)
            documents.append(Document(chunks=chunks, file_name=file))

    with open(os.path.join(SAVE_DIR, EMBEDDED_DOCUMENTS_FILE), "w") as f:
        json.dump([doc.model_dump() for doc in documents], f)



def load_documents():
    with open(os.path.join(SAVE_DIR, EMBEDDED_DOCUMENTS_FILE), "r") as f:
        return [Document(**doc) for doc in json.load(f)]


if __name__ == "__main__":
    embed_documents()