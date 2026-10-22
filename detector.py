"""
detector.py
-----------
Wraps YOLOv8 (via ultralytics) with its built-in ByteTrack tracker so every
detected object gets a *consistent ID* across frames — not just a bounding
box per frame. Without tracking, "the person in the red shirt" would be a
new, disconnected detection every frame and we could never answer questions
like "when did they leave the frame."
"""

from ultralytics import YOLO
import config


class ObjectTracker:
    def __init__(self, weights=config.YOLO_WEIGHTS, conf=config.DETECTION_CONF):
        self.model = YOLO(weights)
        self.conf = conf

    def track_frame(self, frame):
        """
        Run detection + tracking on a single frame.
        Returns a list of dicts: {track_id, class_name, confidence, bbox}
        bbox is (x1, y1, x2, y2) in pixel coordinates.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            tracker=config.TRACKER_CONFIG,
            persist=True,       # remember track IDs between calls
            verbose=False,
        )

        detections = []
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return detections  # nothing tracked this frame (e.g. empty scene)

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for box, track_id, confidence, cls_id in zip(boxes, ids, confs, classes):
            detections.append({
                "track_id": int(track_id),
                "class_name": names[cls_id],
                "confidence": float(confidence),
                "bbox": tuple(float(v) for v in box),
            })
        return detections
