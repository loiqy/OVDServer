from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from PIL import Image
import base64
from io import BytesIO

from .dosod_backend import DOSODBackend

# Parse command line arguments
parser = argparse.ArgumentParser(description="DOSOD model server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
args = parser.parse_args()

model = DOSODBackend()

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
            "classes": ["person", "car"] | "person, car"
        }
        return {
            "result": [
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
        model.load_model(model_name)
        res = model.detect(
            image=image,
            text_prompts=classes,
            confidence_threshold=conf,
            iou_threshold=None
            )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return JSONResponse(content=res, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)