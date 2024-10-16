# PyTorch ViT Image Classification with Apple MPS + Flask APIs + Web Client

This repository demonstrates an end-to-end pipeline for image classification using a Vision Transformer (ViT) model built with PyTorch, optimized for Apple's MPS (Metal Performance Shaders) to leverage GPU acceleration on M1, M2, and M3 Macs. The project includes a Flask-based API for backend model inference, allowing users to interact with the model through a web client. It provides a seamless integration of deep learning, RESTful APIs, and web deployment for real-time image classification tasks.


## Features
- PyTorch-based ViT image classification model, optimized with Apple MPS.
- Flask server with RESTful API for model inference.
- A simple web interface for interacting with the model.
- Supports early stopping during training for efficient training runs.

## Prerequisites
- **Apple Silicon Mac (M1, M2, or M3)** for MPS acceleration (or use CPU).
- Python 3.8 or higher.