from io import BytesIO
from urllib import request
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort

# === Q1
model = onnx.load("hair_classifier_v1.onnx")

output_node_name = [node.name for node in model.graph.output]

print("Q1:", output_node_name)


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# === Q2
img_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
target_size = (200, 200)

img_raw = download_image(img_url)
img = prepare_image(img_raw, target_size)

print("Q2:", img.size)

# === Q3
x = np.array(img)
x = x / 255.0

# normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
x = (x - mean) / std

# the first pixel R
first_pixel_r = x[0, 0, 0]

print(f"Q3: {first_pixel_r:.4f}")


# === Q4

session = ort.InferenceSession("hair_classifier_v1.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input: {input_name}")
print(f"Output: {output_name}")

# Convert for PyTorch format
x = np.transpose(x, (2, 0, 1))

# Add batch dimension from (3, 200, 200) to (1, 3, 200, 200)
x = np.expand_dims(x, axis=0)

# Convert float32
x = x.astype(np.float32)

outputs = session.run([output_name], {input_name: x})

prediction = outputs[0]
print(f"Model output, logit: {prediction}")

# Apply sigmoid
probability = 1 / (1 + np.exp(-prediction))
print(f"Probability: {probability}")

binary_prediction = (probability > 0.5).astype(int)
print(f"Binary prediction: {binary_prediction}")

print("Q4:", probability)


# === Q5
print("Q5: agrigorev/model-2025-hairstyle:v1   9e43d5a5323f        921MB          269MB") # docker pull agrigorev/model-2025-hairstyle:v1 && docker image ls | grep agrigorev


# === Q6
# docker build -t hair-lambda -f Dockerfile .
# docker run -p 9000:8080 hair-lambda
# curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
print("Q6: -0.1022")