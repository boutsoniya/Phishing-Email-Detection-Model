# Phishing Email Detection Model

This project is a Machine Learning pipeline built with Scikit-learn to classify emails as "Phishing" or "Safe" based on their textual content.

## Features
- **Data Loading**: Loads email datasets (CSV format).
- **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert email text into numerical features, capturing important keywords and URL-like patterns.
- **Model Training**: Trains a Multinomial Naive Bayes classifier, which is highly effective for text classification tasks like spam/phishing detection.
- **Evaluation**: Evaluates the model on test data, calculating accuracy and generating a confusion matrix.
- **Interactive Web App**: Includes a Streamlit web interface to interactively test the model with custom email text.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Setup the Environment
Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Usage

#### Train the Model
First, run the training script to train the model on the sample dataset. This will save the trained model to `model.pkl` and the vectorizer to `vectorizer.pkl`.

```bash
python src/train.py
```

#### Evaluate the Model
Run the evaluation script to see the model's accuracy and generate a confusion matrix plot.

```bash
python src/evaluate.py
```

#### Run the Web App
To interactively test the model, launch the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure
- `data/`: Contains the dataset (`sample_dataset.csv`).
- `src/`: Contains the core machine learning scripts.
  - `data_loader.py`: Handles loading and splitting the data.
  - `feature_extraction.py`: Contains logic for TF-IDF vectorization.
  - `train.py`: Script to train the model.
  - `evaluate.py`: Script to evaluate the model's performance.
- `app.py`: Streamlit web application.
- `requirements.txt`: List of Python dependencies.
