# local-rag-test

Testing out how RAG would work with locally hosted models.

## How To

1.  Install [Ollama](https://ollama.com/)
2.  Run the following command to create [python virtual environment](https://docs.python.org/3/library/venv.html) and start it:

        python3 -m venv venv
        source venv/bin/activate

3.  Run the following command to install dependencies:

        pip install -r requirements.txt

4.  Set environment variables inside an `.env` file:
    - `RSS_FEED_PATH`: The path to RSS feed you wan't to use as source for your documents e.f. [NYT](https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml)
    - `COLLECTION_NAME`: The name for [ChromaDB](https://docs.trychroma.com/) collection. You can choose one that is most to your liking e.g. `docs`
    - `EMBEDDING_MODEL`: Name of the embedding model e.g. `mxbai-embed-large`
    - `GENERATIVE_MODEL`: Name of the generative model e.g. `phi3`
5.  Run:

        python src/main.py

6.  When you want to end the conversation then just write `bye`

## Todo

Streamlit UI coming maybe soonish..

## References

https://ollama.com/blog/embedding-models
https://ollama.com/library/phi3
