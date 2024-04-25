import os
import ollama
import click
import chromadb
import feedparser

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

RSS_FEED_PATH = os.getenv("RSS_FEED_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GENERATIVE_MODEL = os.getenv("GENERATIVE_MODEL")

def generate_embedding(doc):
    return ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc)["embedding"]

def pull_model(model):
    current_digest, bars = '', {}
    for progress in ollama.pull(model, stream=True):
        digest = progress.get('digest', '')
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            print(progress.get('status'))
            continue

        if digest not in bars and (total := progress.get('total')):
            bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

        if completed := progress.get('completed'):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest

def init(emod, mod):
    try:
        ollama.show(emod)
    except ollama.ResponseError as e:
        print(f"Error: {e.error}")
        if e.status_code == 404:
            print(f"Pulling the embedding model: {emod}")
            pull_model(emod)

    try:
        ollama.show(mod)
    except ollama.ResponseError as e:
        print(f"Error: {e.error}")
        if e.status_code == 404:
            print(f"Pulling the generative model: {mod}")
            pull_model(mod)

    feed = feedparser.parse(RSS_FEED_PATH)
    feed_entries = feed.entries

    documents = []
    metadatas = []

    for e in tqdm(feed_entries, "Parsing the documents"):
        title = e.title
        link = e.link
        content = e.summary
        tags = ", ".join([t["term"] for t in e.tags])

        documents.append(f"# {title}\n{content}\nTags: {tags}")
        metadatas.append({"title": title, "link": link, "tags": tags})

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = [generate_embedding(d) for d in tqdm(documents, "Generating document embeddings")]
    ids = [f"doc-{i}" for i in range(len(embeddings))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return client, collection

def main_loop(collection):
    while True:
        prompt = input("User:\n")
        print("\n")

        if prompt == "bye":
            break

        result = collection.query(
            query_embeddings=generate_embedding(prompt),
            n_results=3,
        )

        system_task = "Your task is to answer based on The New Yourk Times news feed."
        system_news = "\n\n".join(result["documents"][0])

        stream = ollama.chat(
            model=GENERATIVE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"{system_task} Here might be some news related to the given question: {system_news}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
        )

        print("Assistant: ", flush=True)
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")

def clean(client):
    client.delete_collection(COLLECTION_NAME)

@click.command()
@click.option(
    "--emod",
    prompt="Enter the embedding model name",
    help="The embedding model used for vector retrieval",
    default=EMBEDDING_MODEL
)
@click.option(
    "--mod",
    prompt="Enter the generative model name",
    help="The generative model used for text generation",
    default=GENERATIVE_MODEL
)
def run(emod, mod):
    client, collection = init(emod, mod)
    if collection:
        main_loop(collection)

    if client:
        clean(client)

if __name__ == "__main__":
    try:
        if not RSS_FEED_PATH:
            raise EnvironmentError("RSS path was not defined")
        if not EMBEDDING_MODEL:
            raise EnvironmentError("Embedding model name is not defined")
        if not GENERATIVE_MODEL:
            raise EnvironmentError("Generative model name is not defined")

        run()

    except Exception as e:
        print(e)
