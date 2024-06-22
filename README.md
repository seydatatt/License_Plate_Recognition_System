# License Plate Recognition System

This project is a License Plate Recognition System built using a character dataset and a dataset created from 21 car images. The AI model is trained to recognize license plates from the images. This project was prepared with the help of Umut Kaan Başer's YouTube video.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction
This project aims to develop a License Plate Recognition System using deep learning techniques. The system is trained to identify and read license plates from images of cars. The project utilizes a custom character dataset and images of cars to train the AI model.

## Dataset
The dataset used in this project consists of two parts:
1. **Character Dataset**: A collection of character images used to train the model to recognize individual characters on the license plates.
2. **Car Images**: A dataset of 21 car images with visible license plates used to test and validate the model.

## Model
The AI model is built using a convolutional neural network (CNN) to process the images and recognize the characters on the license plates. The model's architecture and training process are designed to achieve high accuracy in real-world scenarios.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/license-plate-recognition.git
    cd license-plate-recognition
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the datasets:**
    Place the character dataset and car images in the appropriate directories as specified in the code.

## Usage
To use the License Plate Recognition System, follow these steps:

1. **Train the model:**
    Run the training script to train the model with the provided datasets.
    ```sh
    python train_model.py
    ```

2. **Test the model:**
    Use the test script to evaluate the model on the car images.
    ```sh
    python test_model.py
    ```

3. **Recognize license plates:**
    Use the recognition script to recognize license plates from new images.
    ```sh
    python recognize_plate.py --image path/to/image.jpg
    ```

## Acknowledgements
This project was prepared with the help of Umut Kaan Başer's YouTube video. Special thanks to him for providing a detailed tutorial and guidance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
****
