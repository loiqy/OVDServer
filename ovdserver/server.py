import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from PIL import Image
import base64
from io import BytesIO
import traceback

from .backends.base import OVDBackend
import os
import uuid

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(log_dir):
    logger = logging.getLogger("OVDServer")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler_info = FlushFileHandler(f'{log_dir}/info.log')
    f_handler_debug = FlushFileHandler(f'{log_dir}/debug.log')
    f_handler_error = FlushFileHandler(f'{log_dir}/error.log')

    c_handler.setLevel(logging.DEBUG)
    f_handler_info.setLevel(logging.INFO)
    f_handler_debug.setLevel(logging.DEBUG)
    f_handler_error.setLevel(logging.ERROR)

    # Create formatters and add them to handlers
    c_format = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s\n'
        '------------------------------------------------------------'
    )
    f_format = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s\n'
        '------------------------------------------------------------'
    )

    c_handler.setFormatter(c_format)
    f_handler_info.setFormatter(f_format)
    f_handler_debug.setFormatter(f_format)
    f_handler_error.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler_info)
    logger.addHandler(f_handler_debug)
    logger.addHandler(f_handler_error)

    return logger

def create_unique_log_dir(base_dir):
    unique_id = uuid.uuid1().hex  # Use uuid1 to ensure the order is based on creation time
    log_dir = os.path.join(base_dir, unique_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def parse_arguments():
    parser = argparse.ArgumentParser(description="OVD server")
    parser.add_argument("--backend", type=str, default=None, help="Default backend to use")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for logs")
    return parser.parse_args()

logger = None

backend_name = None
model = None

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
                # YOLOWorld/OWLViT/GroundingDINO format
                {
                    'name': 'person',
                    'class': 0,
                    'confidence': 0.91938,
                    'box': {'x1': 669.08563, 'y1': 389.87274, 'x2': 810.0, 'y2': 883.20898}
                },
                # APE format
                {
                    "category_id": 0,
                    "bbox": [
                        360.72900390625,
                        296.6678466796875,
                        82.703857421875,
                        241.85260009765625
                    ],
                    "score": 0.6692016124725342,
                    "segmentation": {
                        "size": [
                            640,
                            640
                        ],
                        "counts": "dWS74:b0bb0@W]Oh0db0>M2X@h..."
                    },
                    "category_name": "person",
                    "image_name": "/path/to/image.jpg"
                },
            ],
            "visualization": base64_encoded_image
        }
    """
    global logger, model, backend_name

    image_data = data.get("image")
    classes = data.get("classes")
    if not image_data:
        logger.error("No image provided")
        return JSONResponse(content={"error": "No image provided"}, status_code=400)

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        logger.debug(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=400)

    if 'backend' in data and data['backend'] != backend_name:
        model = OVDBackend.get_backend(data['backend'])
        if model:
            backend_name = data['backend']
            logger.info(f"Using backend {backend_name}")

    if not model:
        logger.error(f"Invalid backend {model}: {backend_name}")
        return JSONResponse(content={"error": f"Invalid backend {model}: {backend_name}"}, status_code=400)

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
        logger.error(f"Error during detection: {str(e)}")
        logger.debug(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return JSONResponse(content=res, status_code=200)

if __name__ == "__main__":
    args = parse_arguments()
    log_dir = create_unique_log_dir(args.log_dir)
    logger = setup_logger(log_dir)

    backend_name = args.backend
    if backend_name:
        model = OVDBackend.get_backend(backend_name)
        if not model:
            logger.error(f"Invalid backend {backend_name}")
            exit(1)
        else:
            logger.info(f"Using backend {backend_name}")
    uvicorn.run(app, host=args.host, port=args.port)
