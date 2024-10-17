# PyTorch ViT Image Classification with Apple MPS + Flask APIs + Web Client

This repository demonstrates an end-to-end pipeline for image classification using a Vision Transformer (ViT) model built with PyTorch, optimized for Apple's MPS (Metal Performance Shaders) to leverage GPU acceleration on M1, M2, and M3 Macs. The project includes a Flask-based API for backend model inference, allowing users to interact with the model through a web client. It provides a seamless integration of deep learning, RESTful APIs, and web deployment for real-time image classification tasks.

## Table of Contents
1.   [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Project](#running-the-project)
   - [Train the Model](#1-train-the-model)
   - [Run the Flask Server](#2-run-the-flask-server)
5. [Key Topics of Training Model](#key-topics-of-training-model)
   - [Early Stopping](#early-stopping)
   - [Data Augmentation](#data-augmentation)

## Features
- PyTorch-based ViT image classification model, optimized with Apple MPS.
- Flask server with RESTful API for model inference.
- A simple web interface for interacting with the model.

## Prerequisites
- **Apple Silicon Mac (M1, M2, or M3)** for MPS acceleration (or use CPU). Windows GPU (CUDA) support will be added soon.
- Python 3.8 or higher.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/mehradnia/PyTorch-ViT-Image-Classification-with-Apple-MPS-Flask-APIs-Web-Client.git
    cd PyTorch-ViT-Image-Classification-with-Apple-MPS-Flask-APIs-Web-Client
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your dataset:**
    - Create a `data/[YOUR_DATASET]` directory.
    - Add your dataset structured in class-based folders:
      ```
      data/[YOUR_DATASET]/
      ├── class1/
      │   ├── 1.jpg
      │   └── 2.jpg
      ├── class2/
      │   ├── 1.jpg
      │   └── 2.jpg
      ```

      An example for an animals dataset:
      ```
      data/animals/
      ├── cat/
      │   ├── 1.jpg
      │   └── 2.jpg
      ├── dog/
      │   ├── 1.jpg
      │   └── 2.jpg
      ```

## Running the Project

### 1. Train the Model
1. Open the `config.yaml` file and replace `path/to/your/data` with your data directory (eg: `data/animals`).

2. Open the `notebooks/vit_image_classifier.ipynb` file and proceed through the instructions within the notebook to train your model.

3. Once training is completed, you can find the trained model in the `/models` directory.

### 2. Run the Flask Server
1. Start the Flask server using the command:
    ```bash
    python3 app/server.py
    ```

2. Flask will run the server at `http://localhost:8000`.

3. The API exposes:
   - `POST /predict`: Upload an image and get the classification result.

Example POST request:
```bash
curl -X POST -F "file=@path_to_image.jpg" http://localhost:8000/predict
```


## Key Topics of Training Model:

1. ### Early Stopping: 
This technique monitors the model's performance on the validation set during training. If the model's validation loss stops improving for a specified number of epochs (patience), training is halted to prevent overfitting and save time.

2. ### Data Augmentation: 
Random transformations are applied to the training data to increase the variety of the dataset. Techniques like random resizing, rotations, and color jittering are used to help the model generalize better by learning from a broader range of input variations.

3. ### Reproducible Data Splitting:
It refers to the systematic partitioning of datasets into training, validation, and test subsets in a manner that guarantees consistency across multiple runs. This method helps make model evaluations reliable and allows others to replicate the experiments. Reproducible data splitting is important for keeping machine learning workflows fair and ensuring that different models can be compared accurately.