"""
event_store.py
---------------
This is the RAG layer of the project — directly reusing the hybrid-retrieval
experience from the production RAG system, just with video events instead of
document chunks.

Each "event" is a timestamped text record (a caption, or a track
appearing/disappearing). We embed every event's text with a local sentence
embedding model and store the vectors in a FAISS index, so at query time we
can pull only the handful of moments actually relevant to the user's
question instead of stuffing the entire video transcript into the LLM's
context window.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config


class EventStore:
    def __init__(self, embedding_model=config.EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.events = []          # list of dicts: {timestamp, text, track_ids}
        self.index = None         # FAISS index, built once all events are added

    def add_event(self, timestamp, text, track_ids=None):
        self.events.append({
            "timestamp": round(timestamp, 2),
            "text": text,
            "track_ids": track_ids or [],
        })

    def build_index(self):
        """Embed every collected event and build a FAISS flat index over them."""
        if not self.events:
            raise RuntimeError("No events to index — did the pipeline run on the video?")

        texts = [e["text"] for e in self.events]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
        self.index.add(embeddings.astype(np.float32))

    def search(self, query, top_k=config.TOP_K_RETRIEVAL):
        """Return the top_k most relevant events for a natural-language query."""
        if self.index is None:
            raise RuntimeError("Index not built yet — call build_index() first.")

        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            event = dict(self.events[idx])
            event["relevance_score"] = float(score)
            results.append(event)
        return results

    def save(self, index_path=config.INDEX_PATH, events_path=config.EVENTS_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(events_path, "w") as f:
            json.dump(self.events, f, indent=2)

    def load(self, index_path=config.INDEX_PATH, events_path=config.EVENTS_PATH):
        self.index = faiss.read_index(index_path)
        with open(events_path, "r") as f:
            self.events = json.load(f)
