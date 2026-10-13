"""Use the existing PoseTracker estimator with sparse or all-frame audit images."""
import json
import sys
import time
from pathlib import Path
from queue import Full, Queue
from threading import Thread

sys.path.insert(0, '/home/faraz/Code/posetracker')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
from posetracker import tracking  # noqa: E402
from posetracker.calibration import Calibration  # noqa: E402
from posetracker.camera import CameraConfig  # noqa: E402

from scripts.pose_detector import ScaledBoardDetector  # noqa: E402


def main():
    dest = Path(sys.argv[1]).resolve()
    frames = dest / 'frames'
    frames.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    save_all = '--save-all' in sys.argv[2:]
    image_queue = Queue(maxsize=64)
    image_stats = {'saved': 0, 'dropped': 0, 'errors': []}

    def save_images():
        while True:
            item = image_queue.get()
            if item is None:
                return
            name, frame, record = item
            try:
                options = [cv2.IMWRITE_PNG_COMPRESSION, 1] if save_all else []
                if not cv2.imwrite(str(frames / name), frame, options):
                    raise OSError(f'Could not save {name}')
                with (dest / 'images.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                image_stats['saved'] += 1
            except Exception as exc:
                image_stats['errors'].append(str(exc))

    image_thread = Thread(target=save_images, name='pose-image-writer', daemon=True)
    image_thread.start()

    class AuditEstimator(tracking.PoseEstimator):
        previous = 0.
        processed = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.detector = ScaledBoardDetector(self.calibration)

        def estimate(self, frame):
            row = super().estimate(frame)
            self.processed += 1
            now = time.monotonic()
            if save_all or now - self.previous >= .5:
                self.previous = now
                name = f'{self.processed:06d}.png' if save_all else f'{time.monotonic_ns()}.jpg'
                record = {'file': name, 'wall_time': time.time(), 'monotonic': now,
                          'processed_row': self.processed, 'pose': row.copy()}
                try:
                    image_queue.put_nowait((name, frame.copy(), record))
                except Full:
                    image_stats['dropped'] += 1
            return row

    tracking.PoseEstimator = AuditEstimator
    try:
        metadata = tracking.record(CameraConfig(device='/dev/video1', exposure=150),
                        Calibration.load(dest / 'calibration.json'), dest / 'poses.csv', duration=180)
    finally:
        image_queue.put(None)
        image_thread.join(timeout=15)
        image_stats['writer_finished'] = not image_thread.is_alive()
        (dest / 'image-storage.json').write_text(json.dumps(image_stats, indent=2))
    metadata['detector'] = {'roi_xywh': [100, 0, 400, 420], 'internal_scale': 2,
                            'coordinates': 'mapped back to native pixels; original calibration and pose thresholds',
                            'temporal_smoothing': False}
    metadata['stop_reason'] = 'scripted end of acquisition'
    metadata['image_storage'] = {'mode': 'all processed frames, lossless PNG' if save_all else '2 Hz JPEG', **image_stats}
    (dest / 'poses.meta.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
