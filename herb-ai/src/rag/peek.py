# cspell:disable
import os
import sys
from typing import Dict, Any, List, cast
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

CHROMA_DB_DIR: str = "chroma_storage"

def peek_database() -> None:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    print(f"Connecting to Chroma collection inside '{CHROMA_DB_DIR}'...")
    collection = chroma_client.get_collection(name="botanical_knowledge")
    
    total_count = collection.count()
    print(f"Total indexed document chunks: {total_count}\n")
    
    if total_count == 0:
        print("The vector store is currently empty.")
        return
        
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    print("--- PULLING INDEXED DATABASE CONTENT ---")
    
    # FIX: Safely extract and type-cast the lists upfront to prevent Pylance from seeing Optional/None references later
    ids_list: List[str] = results.get("ids") if results.get("ids") is not None else []
    docs_list: List[str] = cast(List[str], results.get("documents")) if results.get("documents") is not None else []
    metadata_list = cast(List[Dict[str, Any]], results.get("metadatas")) if results.get("metadatas") is not None else []
    embeddings_list = cast(List[List[float]], results.get("embeddings")) if results.get("embeddings") is not None else []

    for idx in range(total_count):
        doc_id = ids_list[idx]
        document_text = docs_list[idx] if idx < len(docs_list) else "None"
        
        raw_metadata = metadata_list[idx] if idx < len(metadata_list) else {}
        metadata: Dict[str, Any] = {str(k): v for k, v in raw_metadata.items()}
        
        vector_dimensions = len(embeddings_list[idx]) if idx < len(embeddings_list) else 0
        
        print(f"\n[Item Index {idx + 1}] ID: {doc_id}")
        print(f"-> Metadata: {metadata}")
        print(f"-> Vector Dimensionality: {vector_dimensions} dimensions")
        print(f"-> Extracted Text Content Preview:\n{document_text}")
        print("-" * 40)

if __name__ == "__main__":
    peek_database()