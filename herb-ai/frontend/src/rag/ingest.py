import os
import httpx
import chromadb
import wikipedia

# 1. THE PATH LINK: Perfectly matches query_engine.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../chroma_storage"))


def get_embedding(text: str) -> list[float]:
    # 2. THE MODEL LINK: Must match the query engine's embedding model
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": "nomic-embed-text", "prompt": text}

    try:
        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30.0)
            if response.status_code == 200:
                return response.json().get("embedding", [])
    except Exception as e:
        print(f"❌ Local embedding failed: {e}")
    return []


def ingest_herb(herb_name: str, search_query: str = None):
    query = search_query if search_query else herb_name
    print(f"🔍 Searching Wikipedia for: {query}...")

    try:
        wiki_text = wikipedia.summary(query, sentences=10)
        print("✅ Data found! Generating embeddings...")

        embedding = get_embedding(wiki_text)

        if embedding:
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            # 3. THE COLLECTION LINK: Must match the query engine exactly
            collection = client.get_or_create_collection(name="botanical_knowledge")

            document_text = (
                f"Herb Name: {herb_name}\nClinical & Botanical Data: {wiki_text}"
            )

            collection.add(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[{"source": "wikipedia", "herb": herb_name}],
                ids=[f"wiki_{herb_name.lower().replace(' ', '_')}"],
            )
            print(f"💾 Successfully saved {herb_name} to Chroma database!")
        else:
            print("❌ Failed to generate embedding.")

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Query ambiguous. Options: {e.options[:5]}")
    except wikipedia.exceptions.PageError:
        print(f"❌ No Wikipedia page found for {query}.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Ingesting the herb your vision model identified
    ingest_herb("Doddapatre", search_query="Coleus amboinicus")
