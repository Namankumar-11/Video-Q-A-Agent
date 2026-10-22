"""
Central configuration for the Video Understanding & Q&A Agent.
Keeping every tunable constant here means no magic numbers buried in the
pipeline code, and one place to adjust behavior for a different video/GPU.
"""

import os

# ---- Groq API ----
# Get a free key at https://console.groq.com and set it as an env var:
#   export GROQ_API_KEY="your_key_here"      (Linux/macOS)
#   setx GROQ_API_KEY "your_key_here"        (Windows)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# NOTE: Groq currently has no vision-capable model, so captioning runs
# locally via BLIP (see captioner.py) — no VISION_MODEL config needed.
TEXT_MODEL = "openai/gpt-oss-120b"          # for final Q&A generation

# ---- Object detection / tracking ----
YOLO_WEIGHTS = "yolov8n.pt"      # nano model — fast enough for a 8GB-VRAM laptop GPU
DETECTION_CONF = 0.4             # minimum confidence to keep a detection
TRACKER_CONFIG = "bytetrack.yaml"  # ultralytics' built-in ByteTrack config

# ---- Captioning sampling ----
CAPTION_EVERY_N_SECONDS = 3.0    # how often we ask the VLM to describe the scene
                                  # (captioning every frame is wasteful — the scene
                                  # rarely changes meaningfully faster than this)

# ---- Embedding / retrieval ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, local, no API cost — good enough for
                                        # short event descriptions
TOP_K_RETRIEVAL = 5


OUTPUT_DIR = "output"
INDEX_PATH = os.path.join(OUTPUT_DIR, "event_index.faiss")
EVENTS_PATH = os.path.join(OUTPUT_DIR, "events.json")
