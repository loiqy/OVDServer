from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from PIL import Image
import base64
from io import BytesIO

from .lazy_ape_backend import APEBackend

# Parse command line arguments
parser = argparse.ArgumentParser(description="APE model server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
args = parser.parse_args()

model = APEBackend()

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
                        "counts": "dWS74:b0bb0@W]Oh0db0>M2X@hNd<[1VCjNg<Z1TChNk<\\1RCdNn<c1iB_NW=f1dBZN\\=k1]BWNc=m1WBUNh=R2PBPNP>T2jAmMW>\\2UAnMl>c3N20N200O1OO2000N2000N0200O100N2O1O1M3bNf@UNb=1gBl1b>fNPA[1n>o1K5M3M3M2N2N1O3M1O2N2N3M1O3OOO3M3M3M2O3M3MWJeBR5U=hJQCY5k=oMiA[NS>c1VBYNg=g1[BZNb=b1dB_NZ=`1hBbNT=Z1SCgNi<U1\\CjNf<o0`CROa<h0dCVOe<`0^C^Oe<8dCF^<0lCNX<GPD8R<mNjDP1];cNkD[1o>N3M3M5K3L[Xk3"
                    },
                    "category_name": "person",
                    "image_name": "/home/llq/workspace/dataset/sample_rec/RefCOCOplus/val-00001-of-00002/COCO_train2014_000000008436_5.jpg"
                },...
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