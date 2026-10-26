"""Export YOLOv8n to the ONNX file the YOLO node uses (Windows).

    ..\\ObjectDetection\\robot_env\\Scripts\\python.exe scripts\\export_yolo_onnx.py
    ..\\ObjectDetection\\robot_env\\Scripts\\python.exe scripts\\export_yolo_onnx.py --height 192 --width 320

Writes odse_ws/src/odse_driving/models/yolov8n_<height>x<width>.onnx. Needs
ultralytics and onnx (the ObjectDetection venv has both). The input is fixed
at 384 x 640 by default: the camera image is 640 x 360, so a square 640 x 640
input would spend 40 % of its time on padding. On the Orin's CPU this size
takes 82 ms (640 x 640: 135 ms, 192 x 320: 26 ms, but it misses the ball far
more often; section 9.5 of docs/how_it_works.md).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent.parent / "odse_ws" / "src" / "odse_driving" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parents[2] / "ObjectDetection" / "yolov8n.pt"))
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "yolov8n.pt"  # export writes next to the weights
        shutil.copy(args.weights, weights)
        path = YOLO(str(weights)).export(format="onnx", imgsz=(args.height, args.width), opset=17, simplify=True)
        out = MODELS / f"yolov8n_{args.height}x{args.width}.onnx"
        shutil.move(path, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
