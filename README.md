# SmartSuggest: Session-Based Recommender System

SmartSuggest is a session-based recommender system built using GRU (Gated Recurrent Units) and attention mechanisms. The system predicts the next item a user may be interested in based on their previous interactions (session data). It is designed for e-commerce websites to offer personalized recommendations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Setup and Model Upload](#setup-and-model-upload)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Model Information](#model-information)
7. [License](#license)

## Project Overview

SmartSuggest leverages deep learning techniques like GRU and Attention to provide session-based recommendations. It uses sequential user interaction data to predict the next item that a user might engage with, offering personalized product recommendations.

Key features:
- Session-based recommendation system using GRU with attention mechanism.
- Supports item and category encoding for efficient processing.
- Built using PyTorch and Streamlit for easy deployment.
- Model hosted on Hugging Face Hub for remote access and inference.

## Installation

### Install Dependencies

Clone the repository and install the required Python libraries by running the following steps:

1. Clone the repository:
    - Go to your terminal and run the command:
      ```
      git clone https://github.com/yourusername/smart-suggest.git
      cd smart-suggest
      ```

2. Install the dependencies:
    ```
    pip install streamlit torch huggingface_hub joblib
    ```

### Setup and Model Upload

To run the recommender system, you need to upload the trained model (`best_smartsuggest_model.pt`) to Hugging Face Hub.

1. **Create a Hugging Face account** if you don't already have one: [Hugging Face Signup](https://huggingface.co).
2. **Create a new model repository** on Hugging Face Hub to upload the model file.
3. **Upload the model** using Git Large File Storage (LFS):
    - Ensure you have `git-lfs` installed by running:
      ```
      git lfs install
      ```

4. **Login to Hugging Face** and clone the model repository:
    ```
    huggingface-cli login
    git clone https://huggingface.co/yourusername/model_name
    cd model_name
    git lfs track "*.pt"
    git add best_smartsuggest_model.pt
    git commit -m "Upload trained model"
    git push
    ```

This will upload the `best_smartsuggest_model.pt` to your Hugging Face model repository, which will allow you to load the model remotely.

### Model File Upload

If your model file (`best_smartsuggest_model.pt`) is large and cannot be uploaded directly to GitHub, use Hugging Face's Git LFS to manage large files.

## Usage

### Running the Streamlit App

To run the Streamlit app, follow these steps:

1. Ensure you have uploaded the model to Hugging Face as described in the "Setup and Model Upload" section.
2. Run the following command to start the Streamlit app:
    ```
    streamlit run app.py
    ```

This will start the Streamlit web app, and you can access it in your browser at `http://localhost:8501` (or wherever Streamlit is hosted).

### How it Works

- The app allows users to interact with the model by uploading e-commerce session data (e.g., views of items).
- The model processes the session data, predicting the next item that the user is likely to interact with based on previous interactions.
- The app displays the top recommendations for a given session, offering a personalized shopping experience.

## Project Structure

The project directory is structured as follows:

smart-suggest/
├── app.py # Main application file (Streamlit app)
├── model.py # Model definition and loading
├── utils.py # Utility functions for data processing
├── requirements.txt # List of required dependencies
├── models/
│ ├── item_encoder.pkl # Item encoder for label encoding
│ └── category_encoder.pkl # Category encoder for label encoding
└── README.md # Project documentation


- **app.py**: This is the main Streamlit application file where the user interacts with the recommender system.
- **model.py**: Contains the model architecture (GRU + Attention) and code for loading the model from Hugging Face Hub.
- **utils.py**: Includes utility functions for processing the dataset and preparing inputs for the model.
- **requirements.txt**: A list of Python dependencies required to run the project.
- **models/**: Directory containing the pre-trained `item_encoder.pkl` and `category_encoder.pkl` files.

## Model Information

The model used in this project is a session-based recommender system built using the following components:

- **GRU (Gated Recurrent Units)**: A type of recurrent neural network (RNN) used for processing sequential data.
- **Attention Mechanism**: Helps focus on the most relevant parts of the sequence for predicting the next item.
- **Cross-Entropy Loss**: Used for training the model with label smoothing to avoid overfitting.

### How the Model Works

The model is trained to predict the next item in a user session. It takes as input a sequence of items and categories that the user has interacted with in the past. Using the GRU, it processes this sequence and generates a probability distribution over all possible items, ranking them by likelihood. The attention mechanism allows the model to focus on the most relevant items in the session, improving the recommendation quality.

### Loading the Model from Hugging Face Hub

To load the model from Hugging Face Hub, you can use the `huggingface_hub` library as shown in the `app.py` file:

```python
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(repo_id="yourusername/model_name", filename="best_smartsuggest_model.pt")
model.load_state_dict(torch.load(MODEL_PATH))
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
This `README.md` file combines the project overview, installation steps, setup instructions for uploading the model, usage guide, and project structure. It also includes the necessary steps for uploading large model files to Hugging Face and loading them for inference.
