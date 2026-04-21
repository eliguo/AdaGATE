### Index a stress-test variant into a Pinecone namespace

import os
import re
import json
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

PINECONE_INDEX = "seal-v3-hard"
EMBED_MODEL    = "text-embedding-3-small"
N              = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="path to variant JSON file")
    parser.add_argument("--namespace", required=True, help="pinecone namespace for this variant")
    args = parser.parse_args()

    print(f"loading data from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    data = data[:N]
    print(f"indexing {len(data)} examples into namespace: {args.namespace}")

    # Build document chunks, strip noise suffixes like [natural-redundant-N]
    documents = []
    for example in tqdm(data, desc="building documents"):
        example_id = example["_id"]
        for item in example["context"]:
            title      = re.sub(r'\s*\[.*?\]', '', item[0]).strip()
            chunk_text = f"{title}: " + " ".join(item[1])
            documents.append(Document(
                page_content=chunk_text,
                metadata={"example_id": example_id, "title": title}
            ))

    print(f"total documents: {len(documents)}")
    print(f"uploading to index={PINECONE_INDEX} namespace={args.namespace}...")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    PineconeVectorStore.from_documents(
        documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
        namespace=args.namespace,
    )
    print(f"done, namespace '{args.namespace}' indexed successfully")


if __name__ == "__main__":
    main()