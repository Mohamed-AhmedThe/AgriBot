from __future__ import annotations

import os
from datetime import datetime
from typing import List

import chromadb
from chromadb.utils import embedding_functions

class MemoryNode:
    def __init__(self, db_path: str = "memory_store") -> None:
        """
        Initializes the Persistent Vector Database.
        """
        print(f"[MemoryNode] Initializing Persistent ChromaDB at ./{db_path}...")
        
        # Ensure the storage directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # PersistentClient ensures memories survive server restarts
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Local embedding model (~90MB, downloads once automatically on first run)
        # Fast, free, and completely offline.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or fetch the agricultural memory collection
        self.collection = self.client.get_or_create_collection(
            name="agribot_event_log",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Cosine similarity is optimal for semantic text search
        )

    def store(self, event_text: str) -> str:
        """
        API Contract: Saves a session event into long-term memory.
        Used by the Supervisor to log decisions and hardware actions.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc_id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        try:
            self.collection.add(
                documents=[event_text],
                metadatas=[{"timestamp": timestamp}],
                ids=[doc_id]
            )
            print(f"[MemoryNode] Logged event: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"[MemoryNode Error] Failed to store event: {str(e)}")
            return None

    def recall(self, query: str, k: int = 3) -> List[str]:
        """
        API Contract: Retrieves the top 'k' most relevant past events based on meaning.
        Used by the Supervisor to inject past context into the LLM prompt.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            memories: List[str] = []

            # ChromaDB returns nested lists for batched queries
            if results.get('documents') and len(results['documents']) > 0:
                docs = results['documents'][0]
                metas = results['metadatas'][0]

                for doc, meta in zip(docs, metas):
                    time = meta.get("timestamp", "Unknown Time")
                    memories.append(f"[{time}]: {doc}")

            return memories

        except Exception as e:
            print(f"[MemoryNode Error] Recall query failed: {str(e)}")
            return []

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """
        Alias for recall() matching the Supervisor's call signature:
            self.memory.query(fn_args["query"], n_results=fn_args.get("n_results", 3))
        """
        return self.recall(query_text, k=n_results)