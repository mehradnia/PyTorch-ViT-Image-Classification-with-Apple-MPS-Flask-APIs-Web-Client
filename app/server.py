# app.py
import torch
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image
from timm import create_model
from flask_cors import CORS  # Import CORS
import yaml

with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Define the Flask app
app = Flask(__name__, static_folder='static',)
CORS(app)

# Load the saved model
# Change num_classes based on your dataset
model = create_model(config['pre_trained_model'], num_classes=90)
model.load_state_dict(torch.load(
    f"../models/{config['APP_MODEL']}.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")

    # Apply the transforms
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    # Return the prediction as a JSON response
    return jsonify({"prediction": predicted.item()})


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


if __name__ == "__main__":
    app.run(debug=True, port=config['PORT'])
