"""
main.py
-------
Entry point. Two modes:
  python main.py process path/to/video.mp4      -> builds the event index
  python main.py ask                             -> loads a saved index and
                                                     answers questions interactively
Kept as two separate commands (rather than always reprocessing) because
detection+tracking+captioning a video is the expensive step — you shouldn't
have to redo it every time you want to ask a new question.
"""

import sys
import os
from pipeline import VideoQAPipeline
from event_store import EventStore
from query_engine import VideoQueryEngine
import config


def cmd_process(video_path):
    if not os.path.exists(video_path):
        print(f"Error: video not found at {video_path}")
        sys.exit(1)
    pipeline = VideoQAPipeline(video_path)
    pipeline.process()
    print(f"\nIndex saved to {config.INDEX_PATH} — run `python main.py ask` to query it.")


def cmd_ask():
    if not os.path.exists(config.INDEX_PATH):
        print("No saved index found. Run `python main.py process <video_path>` first.")
        sys.exit(1)

    store = EventStore()
    store.load()
    engine = VideoQueryEngine(store)

    print(f"Loaded {len(store.events)} events. Ask questions about the video "
          f"(type 'exit' to quit).\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = engine.answer(question)
        print(f"\nA: {result['answer']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python main.py process <video_path>\n  python main.py ask")
        sys.exit(1)

    command = sys.argv[1]
    if command == "process":
        if len(sys.argv) < 3:
            print("Usage: python main.py process <video_path>")
            sys.exit(1)
        cmd_process(sys.argv[2])
    elif command == "ask":
        cmd_ask()
    else:
        print(f"Unknown command: {command}")
        print("Usage:\n  python main.py process <video_path>\n  python main.py ask")
