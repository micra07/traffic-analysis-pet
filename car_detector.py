import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import json
import time


class CarDetector:
    """Детекция автомобилей с использованием YOLOv8."""

    TARGET_CLASSES = [2, 3, 7]
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, model_name: str = 'yolov8n.pt'):
        self.model = YOLO(model_name)
        self.track_history = defaultdict(list)
        self.confidence_threshold = 0.25

    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        results = self.model.predict(
            frame,
            classes=self.TARGET_CLASSES,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.model.names[cls]
                })

        return detections

    def detect_with_tracking(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        results = self.model.track(
            frame,
            classes=self.TARGET_CLASSES,
            conf=self.confidence_threshold,
            persist=True,
            verbose=False
        )

        tracked_objects = []

        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy
                ids = result.boxes.id
                confs = result.boxes.conf

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.tolist()
                    track_id = int(ids[i])
                    conf = float(confs[i])

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    self.track_history[track_id].append((center_x, center_y))

                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)

                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'trajectory': self.track_history[track_id].copy()
                    })

        return tracked_objects

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def draw_tracking(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = obj['bbox']
            trajectory = obj['trajectory']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i - 1]
                    pt2 = trajectory[i]
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

    def process_video(self, video_path: str, show_video: bool = False) -> Dict:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        stats = {
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'total_detections': 0,
            'unique_tracks': 0,
            'frames_processed': 0
        }

        frame_idx = 0
        start_time = time.time()
        last_report_time = start_time

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracked_objects = self.detect_with_tracking(frame, frame_idx)

            stats['total_detections'] += len(tracked_objects)
            stats['frames_processed'] += 1

            if show_video:
                frame = self.draw_tracking(frame, tracked_objects)

                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Car Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

            current_time = time.time()
            if current_time - last_report_time >= 60:
                elapsed = current_time - start_time
                print(f"  Progress: {frame_idx}/{total_frames} frames ({elapsed:.0f}s elapsed)")
                last_report_time = current_time

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        stats['unique_tracks'] = len(self.track_history)

        return stats

    def process_folder(self, input_folder: str, output_folder: str) -> None:
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        output_path.mkdir(parents=True, exist_ok=True)

        video_files = []
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                video_files.append(file_path)

        if not video_files:
            print(f"No video files found in {input_folder}")
            return

        print(f"Found {len(video_files)} video(s)")
        print("=" * 60)

        all_stats = []

        for video_file in video_files:
            print(f"\nFile: {video_file.name}")

            try:
                print(f"  [PROCESS] Detecting cars...")
                stats = self.process_video(str(video_file), show_video=False)

                stats_filename = f"{video_file.stem}_stats.json"
                stats_path = output_path / stats_filename

                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                print(f"  [SUCCESS] Detections: {stats['total_detections']}, "
                      f"Unique tracks: {stats['unique_tracks']}")
                print(f"  [SAVE] Stats saved to: {stats_path}")

                all_stats.append(stats)

            except Exception as e:
                print(f"  [ERROR] {str(e)}")

        print("\n" + "=" * 60)
        print(f"Completed: {len(all_stats)} videos processed")
        print(f"Output folder: {output_path.absolute()}")
        print("=" * 60)

        if all_stats:
            total_detections = sum(s['total_detections'] for s in all_stats)
            total_tracks = sum(s['unique_tracks'] for s in all_stats)
            print(f"\nTotal detections: {total_detections}")
            print(f"Total unique tracks: {total_tracks}")


if __name__ == "__main__":
    detector = CarDetector()
    detector.process_folder("materials_with_watermark", "detection_results")