from ..base import OVDBackend
import base64
import json
from PIL import Image
from pathlib import Path
from io import BytesIO
from typing import List, Union

class DOSODBackend(OVDBackend):
    model_names = [
        'owlvit-base-patch16',
        'owlvit-base-patch32',
        'owlvit-large-patch14',
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
        self.model = OWLViT(model_name + ".pt")
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