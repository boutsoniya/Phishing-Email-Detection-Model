import os
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_loader import load_data, split_data
from feature_extraction import extract_features

def main():
    print("--- Starting Evaluation Process ---")
    
    # 1. Load the trained model and vectorizer
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Model or vectorizer not found. Please run 'python src/train.py' first.")
        return
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
    
    # 2. Load and Split Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset.csv')
    df = load_data(data_path)
    if df is None: return
    
    # We only need the test set for evaluation
    _, X_test, _, y_test = split_data(df)
    
    # 3. Transform test data using the loaded vectorizer
    # We only pass X_test here so it doesn't refit
    X_test_features = vectorizer.transform(X_test)
    
    # 4. Make Predictions
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test_features)
    
    # 5. Evaluate Performance
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)']))
    
    # 6. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Phishing'], 
                yticklabels=['Safe', 'Phishing'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")

if __name__ == "__main__":
    main()
