from ..base import OVDBackend
import base64
from PIL import Image
from io import BytesIO
from typing import List, Union

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import numpy as np

class OWLv2Backend(OVDBackend):
    model_names = [
        'owlv2-base-patch16-ensemble',
        'owlv2-large-patch14-ensemble',
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"google/{model_name}"
        self.processor: Owlv2Processor = Owlv2Processor.from_pretrained(model_id)
        self.model: Owlv2ForObjectDetection = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        # self.model = self.model.to("cuda")
        self.model_name = model_name

    def detect(
            self,
            image: Image.Image,
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

        texts = [[f'a photo of {class_name}' for class_name in classes]]
        try:
            inputs = self.processor(text=texts, images=image, return_tensors="pt", truncation=True).to(self.device)
            # inputs.to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e1:
            try:
                image = image.convert("RGB")
                inputs = self.processor(images=image, text=texts, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except Exception as e2:
                raise ValueError(f"Failed to process image on first attempt: {e1}. Failed on second attempt: {e2}") from e2

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        target_sizes = target_sizes.to("cuda")
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=confidence_threshold, target_sizes=target_sizes)

        text = classes
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

        instances = []
        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            instances.append({
                "name": text[label.item()],
                "class": label.item(),
                "confidence": round(score.item(), 5),
                "box": {
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3]
                }
            })

        # Generate random saturated colors for each class
        colors = {}
        for label in text:
            hsv_color = np.array([np.random.randint(0, 180), 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
            colors[label] = tuple(int(c) for c in rgb_color)

        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]
            class_name = text[label]
            color = colors.get(class_name, (255, 255, 255))  # Default to white if class not in colors

            # Draw the bounding box
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw the label with background
            label_size, base_line = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (box[0], box[1] - label_size[1] - base_line)
            bottom_right = (box[0] + label_size[0], box[1])
            cv2.rectangle(cv_image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(cv_image, class_name, (box[0], box[1] - base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Convert back to PIL image
        visualization = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        visualization.save(buffered, format="PNG")
        visualization_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"result": instances, "visualization": visualization_base64}