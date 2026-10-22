"""
pipeline.py
-----------
Orchestrates the whole ingestion process: read the video frame by frame,
run detection+tracking on every frame (cheap, local, no API calls), and
periodically (every N seconds) send a frame to the VLM for a natural-
language caption (expensive, API call — hence the sampling instead of
doing it every frame).

Two kinds of events get logged:
  1. Track lifecycle events — "a person appeared" / "a person left the frame"
     (derived purely from the tracker, free, no API needed)
  2. Scene captions — richer natural-language descriptions (from the VLM,
     sampled periodically)

Both get pushed into the same EventStore so retrieval can pull from either.
"""

import cv2
from detector import ObjectTracker
from captioner import SceneCaptioner
from event_store import EventStore
import config


class VideoQAPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.tracker = ObjectTracker()
        self.captioner = SceneCaptioner()
        self.event_store = EventStore()

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        caption_interval_frames = int(fps * config.CAPTION_EVERY_N_SECONDS)

        seen_tracks = {}   # track_id -> class_name, to detect appear/disappear events
        active_this_frame = set()
        frame_idx = 0

        print(f"Processing video at {fps:.1f} fps, "
              f"captioning every {config.CAPTION_EVERY_N_SECONDS}s...")

        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp = frame_idx / fps
            detections = self.tracker.track_frame(frame)
            active_this_frame = {d["track_id"] for d in detections}

            # ---- track lifecycle events (appear/disappear) ----
            for d in detections:
                if d["track_id"] not in seen_tracks:
                    seen_tracks[d["track_id"]] = d["class_name"]
                    self.event_store.add_event(
                        timestamp,
                        f"A {d['class_name']} (track #{d['track_id']}) entered the frame.",
                        track_ids=[d["track_id"]],
                    )

            disappeared = set(seen_tracks.keys()) - active_this_frame
            for track_id in list(disappeared):
                class_name = seen_tracks.pop(track_id, "object")
                self.event_store.add_event(
                    timestamp,
                    f"The {class_name} (track #{track_id}) left the frame.",
                    track_ids=[track_id],
                )

            # ---- periodic VLM captioning ----
            if frame_idx % caption_interval_frames == 0:
                active_classes = [d["class_name"] for d in detections]
                caption = self.captioner.caption_frame(frame, active_classes)
                self.event_store.add_event(
                    timestamp,
                    caption,
                    track_ids=list(active_this_frame),
                )
                print(f"  [{timestamp:6.1f}s] {caption}")

            frame_idx += 1

        cap.release()
        print(f"Done. Logged {len(self.event_store.events)} events from {frame_idx} frames.")

        self.event_store.build_index()
        self.event_store.save()
        return self.event_store
