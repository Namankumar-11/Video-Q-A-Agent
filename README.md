# Video Understanding & Q&A Agent

Ask natural-language questions about a video ("when did the person in red enter?",
"summarize what happened between 1:00 and 2:00") and get grounded, timestamp-cited
answers — built by combining object detection/tracking, VLM scene captioning, and
retrieval-augmented generation.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # free key at https://console.groq.com
```

First run will auto-download `yolov8n.pt` (~6MB) from Ultralytics.

## Usage

```bash
# Step 1: process a video into a searchable event index (one-time cost)
python main.py process path/to/your_video.mp4

# Step 2: ask questions about it (fast — no reprocessing needed)
python main.py ask
```

## How it works

1. **Detection + tracking** (`detector.py`) — YOLOv8 + ByteTrack find objects and
   give each one a persistent ID across frames.
2. **Scene captioning** (`captioner.py`) — every few seconds, a sampled frame is
   sent to a vision-language model for a natural-language description.
3. **Event storage + retrieval** (`event_store.py`) — every caption and track
   appear/disappear event is embedded and stored in a FAISS index.
4. **Query answering** (`query_engine.py`) — your question retrieves the most
   relevant events, which get passed to an LLM that must cite timestamps and
   isn't allowed to answer from outside knowledge.

## Known limitations

- Captioning cost scales with video length — a 10-minute video at the default
  3-second interval makes ~200 VLM API calls.
- Track IDs can drift if an object is fully occluded for a long time (a
  limitation of ByteTrack, not this project).
- No audio/speech understanding yet — purely visual.
