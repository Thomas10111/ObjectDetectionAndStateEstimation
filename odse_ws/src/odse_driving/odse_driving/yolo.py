"""YOLOv8 object detection with ONNX Runtime, without ultralytics or PyTorch.

Only numpy, OpenCV and onnxruntime, which the Orin container has. The model is
an ultralytics ONNX export (scripts/export_yolo_onnx.py) with a fixed input
size; the image is letterboxed into it as ultralytics does. Output boxes are
in pixels of the original image. Used by yolo_node.py (M2 of
docs/interfaces.md).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# The 80 COCO classes in the order YOLOv8 was trained on
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


@dataclass
class Box:
    class_id: str
    score: float
    u: float  # center, pixels from the left
    v: float  # center, pixels from the top
    width: float
    height: float


class Yolo:
    def __init__(self, model_path: str, threads: int = 0) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0: onnxruntime decides
        options.log_severity_level = 3
        # Plain CPU provider: on the Orin it is about twice as fast as XNNPACK
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_h, self.input_w = model_input.shape[2], model_input.shape[3]

    def detect(self, rgb: np.ndarray, conf: float = 0.5, iou: float = 0.45) -> list[Box]:
        """All boxes with score >= conf in an RGB image (H x W x 3, uint8)."""
        blob, scale, pad_x, pad_y = self._letterbox(rgb)
        # (1, 84, N): per candidate 4 box values (cx, cy, w, h) then 80 class scores
        out = self.session.run(None, {self.input_name: blob})[0][0].T
        scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= conf
        if not keep.any():
            return []
        boxes, best, cls = out[keep, :4], best[keep], cls[keep]
        # back to pixels of the original image
        boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
        boxes[:, 2:] /= scale
        result = []
        for c in np.unique(cls):  # non maximum suppression per class
            idx = np.flatnonzero(cls == c)
            xywh = [[float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])] for b in boxes[idx]]
            for k in np.array(cv2.dnn.NMSBoxes(xywh, best[idx].tolist(), conf, iou)).flatten():
                b = boxes[idx[k]]
                result.append(Box(COCO_CLASSES[c], float(best[idx[k]]), *map(float, b)))
        return sorted(result, key=lambda b: -b.score)

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Scale to fit the model input keeping the aspect ratio, pad with gray (114)."""
        h, w = rgb.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (self.input_w - new_w) / 2, (self.input_h - new_h) / 2
        canvas = np.full((self.input_h, self.input_w, 3), 114, np.uint8)
        top, left = round(pad_y - 0.1), round(pad_x - 0.1)
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, left, top
