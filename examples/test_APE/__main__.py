import base64
import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="APE Test Script")
    parser.add_argument("--host", type=str, default="localhost", help="Host address of the server")
    parser.add_argument("--port", type=int, default=8081, help="Port number of the server")
    return parser.parse_args()

args = parse_args()
host = args.host
port = args.port
url = f"http://{host}:{port}/predict/"

def test_predict_no_image():
    response = requests.post(url, json={"classes": ["person", "car"]})
    assert response.status_code == 400
    assert response.json() == {"error": "No image provided"}

def test_predict_invalid_image():
    response = requests.post(url, json={"image": "invalid_base64", "classes": ["person", "car"]})
    assert response.status_code == 400
    assert "error" in response.json()

def test_predict_valid_image():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"image": image_data, "classes": ["person", "car"]})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json())

def test_predict_with_model_name():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"image": image_data, "classes": ["person", "car"], "model_name": "APE_D"})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json())

def test_predict_with_str_classes():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"image": image_data, "classes": "person, bus"})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json())

if __name__ == "__main__":
    test_predict_no_image()
    test_predict_invalid_image()
    test_predict_valid_image()
    test_predict_with_model_name()
    test_predict_with_str_classes()
    print("All tests passed!")