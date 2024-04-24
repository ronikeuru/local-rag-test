import ollama
import chromadb
import feedparser

collection_name = "docs"

feed = feedparser.parse("https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml")
feed_entries = feed.entries

documents = []
metadatas = []

for e in feed_entries:
    title = e.title
    link = e.link
    content = e.summary
    tags = ", ".join([t["term"] for t in e.tags])

    documents.append(f"# {title}\n{content}\nTags: {tags}")
    metadatas.append({"title": title, "link": link, "tags": tags})

client = chromadb.Client()
collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

def generate_embedding(doc):
    return ollama.embeddings(model="mxbai-embed-large", prompt=doc)["embedding"]

embeddings = [generate_embedding(d) for d in documents]
ids = [f"doc-{i}" for i in range(len(embeddings))]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)

prompt = "Is there any news about Google?"

result = collection.query(
    query_embeddings=generate_embedding(prompt),
    n_results=3,
)

data = result["documents"][0]

system_task = "Your task is to answer based on The New Yourk Times news feed."
system_news = "\n\n".join(data)

stream = ollama.chat(
    model="phi3",
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

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

client.delete_collection(collection_name)