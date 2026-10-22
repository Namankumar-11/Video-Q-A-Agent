"""
query_engine.py
----------------
Takes a user's natural-language question, retrieves the most relevant
timestamped events from the EventStore, and asks the LLM to answer using
ONLY those events — with explicit timestamp citations. This mirrors the
"citation grounding verification" approach from the production RAG system:
we never let the model answer from its own general knowledge about what
"might" be in a video; it must point back to retrieved evidence.
"""

from groq import Groq
import config


class VideoQueryEngine:
    def __init__(self, event_store, api_key=config.GROQ_API_KEY, model=config.TEXT_MODEL):
        self.event_store = event_store
        self.client = Groq(api_key=api_key)
        self.model = model

    def _format_context(self, retrieved_events):
        lines = []
        for e in retrieved_events:
            mins, secs = divmod(e["timestamp"], 60)
            timestamp_str = f"{int(mins):02d}:{secs:05.2f}"
            lines.append(f"[{timestamp_str}] {e['text']}")
        return "\n".join(lines)

    def answer(self, question, top_k=config.TOP_K_RETRIEVAL):
        retrieved = self.event_store.search(question, top_k=top_k)

        if not retrieved:
            return {
                "answer": "I couldn't find anything relevant to that question in the video.",
                "sources": [],
            }

        context = self._format_context(retrieved)

        prompt = f"""You are answering questions about a video using only the timestamped
event log below. Every line is something that was actually detected or observed
in the video at that timestamp.

EVENT LOG:
{context}

QUESTION: {question}

Answer using ONLY the information in the event log above. Cite the specific
timestamp(s) you used, in [MM:SS] format. If the event log doesn't contain
enough information to answer confidently, say so explicitly instead of guessing.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # keep answers grounded, not creative
            max_tokens=400,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": retrieved,
        }
