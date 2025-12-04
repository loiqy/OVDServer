from ..base import OVDBackend
from ultralytics import YOLOWorld
from ultralytics.engine.results import Results
from PIL import Image
from pathlib import Path
import cv2
import base64
from io import BytesIO
import json
from typing import List, Union

class YOLOWorldBackend(OVDBackend):
    model_names = [
        'yolov8s-world',
        'yolov8m-world',
        'yolov8l-world',
        'yolov8x-world',
        'yolov8s-worldv2',
        'yolov8m-worldv2',
        'yolov8l-worldv2',
        'yolov8x-worldv2'
        ]
    def __init__(self, model_name=None):
        self.model = None
        if model_name is None:
            self.model_name =  self.model_names[-1]
        else:
            self.load_model(model_name)

    def load_model(self, model_name: str):
        if model_name not in self.model_names:
            raise ValueError("Invalid model name")
        if self.model is not None and model_name == self.model_name:
            return
        self.model = YOLOWorld(model_name + ".pt")
        self.model_name = model_name

    def detect(
            self,
            image: Union[str, Path, Image.Image],
            text_prompts: Union[str, List[str]],
            confidence_threshold: float = 0.05,
            iou_threshold: float = 1.0,
            ):
        if self.model is None:
            self.load_model(self.model_name)
        if isinstance(text_prompts, str):
            classes = [c.strip() for c in text_prompts.split(",")]
        elif isinstance(text_prompts, list):
            classes = [c.strip() for c in text_prompts]
        else:
            raise ValueError("Invalid text_prompts format")
        self.model.set_classes(classes)
        results: List[Results] = self.model.predict(source=image, conf=confidence_threshold, iou=iou_threshold)
        visualization = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        visualization.save(buffered, format="PNG")
        visualization_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result = json.loads(results[0].to_json(normalize=False, decimals=5))
        return {"result": result, "visualization": visualization_base64}
    