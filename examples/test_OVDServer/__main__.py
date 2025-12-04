import base64
import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="OVDServer Test Script")
    parser.add_argument("--backend", type=str, default=None, help="Default backend to use")
    parser.add_argument("--model_name", type=str, default=None, help="Model name to use")
    parser.add_argument("--host", type=str, default="localhost", help="Host address of the server")
    parser.add_argument("--port", type=int, default=8081, help="Port number of the server")
    return parser.parse_args()

args = parse_args()
backend = args.backend
model_name = args.model_name
host = args.host
port = args.port
url = f"http://{host}:{port}/predict/"

def save_visualization(visualization_base64, filename):
    with open(filename, "wb") as image_file:
        image_file.write(base64.b64decode(visualization_base64))

def test_predict_no_image():
    response = requests.post(url, json={"backend": backend, "classes": ["person", "car"]})
    assert response.status_code == 400
    assert response.json() == {"error": "No image provided"}

def test_predict_invalid_image():
    response = requests.post(url, json={"backend": backend, "image": "invalid_base64", "classes": ["person", "car"]})
    assert response.status_code == 400
    assert "error" in response.json()

def test_predict_valid_image():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"backend": backend, "image": image_data, "classes": ["person", "car"]})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json()['result'])
    save_visualization(response.json()["visualization"], "test_predict_valid_image.png")

def test_predict_with_model_name():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"backend": backend, "image": image_data, "classes": ["person", "car"], "model_name": model_name})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json()['result'])
    save_visualization(response.json()["visualization"], "test_predict_with_model_name.png")

def test_predict_with_str_classes():
    with open("assets/bus.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = requests.post(url, json={"backend": backend, "image": image_data, "classes": "person, bus"})
    if 'error' in response.json():
        print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()
    print(response.json()['result'])
    save_visualization(response.json()["visualization"], "test_predict_with_str_classes.png")

if __name__ == "__main__":
    test_predict_no_image()
    test_predict_invalid_image()
    test_predict_valid_image()
    test_predict_with_model_name()
    test_predict_with_str_classes()
    print("All tests passed!")