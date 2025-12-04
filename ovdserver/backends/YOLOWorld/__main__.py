from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from PIL import Image
import base64
from io import BytesIO

from .yoloworld_backend import YOLOWorldBackend

# Parse command line arguments
parser = argparse.ArgumentParser(description="YOLO-World model server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
args = parser.parse_args()

model = YOLOWorldBackend()

app = FastAPI()

@app.post("/predict/")
async def predict(data: dict):
    """
    Endpoint for predicting on an image

    Args:
        data (dict): A dictionary containing the image data and classes

    Returns:
        JSONResponse: A JSON response containing the prediction results

    Example:
        data = {
            "image": "base64_encoded_image",
            "classes": ["person", "car"]
        }
        return {
            "result": [
                {
                    'name': 'person',
                    'class': 0,
                    'confidence': 0.91938,
                    'box': {'x1': 669.08563, 'y1': 389.87274, 'x2': 810.0, 'y2': 883.20898}
                },
                {
                    'name': 'person',
                    'class': 0,
                    'confidence': 0.918,
                    'box': {'x1': 50.5752, 'y1': 400.56348, 'x2': 248.22014, 'y2': 901.3302}
                },
                {
                    'name': 'person',
                    'class': 0,
                    'confidence': 0.91697,
                    'box': {'x1': 222.88832, 'y1': 405.75104, 'x2': 345.39194, 'y2': 859.51477}
                },
                {
                    'name': 'person',
                    'class': 0,
                    'confidence': 0.76346,
                    'box': {'x1': 0.0, 'y1': 411.74319, 'x2': 78.96465, 'y2': 1075.38989}
                }
            ],
            "visualization": base64_encoded_image
        }
    """
    global model

    image_data = data.get("image")
    classes = data.get("classes")
    if not image_data:
        return JSONResponse(content={"error": "No image provided"}, status_code=400)

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    model_name = model.model_name
    if "model_name" in data:
        model_name = data.get("model_name")

    try:
        conf = data.get("conf", 0.05)
        iou = data.get("iou", 1.0)
        model.load_model(model_name)
        res = model.detect(
            image=image,
            text_prompts=classes,
            confidence_threshold=conf,
            iou_threshold=iou
            )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return JSONResponse(content=res, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)