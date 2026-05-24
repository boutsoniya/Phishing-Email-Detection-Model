import os
# pyrefly: ignore [missing-import]
import joblib
from sklearn.naive_bayes import MultinomialNB
from data_loader import load_data, split_data
from feature_extraction import get_vectorizer, extract_features

def main():
    print("--- Starting Training Process ---")
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset.csv')
    df = load_data(data_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
        
    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Extract Features
    vectorizer = get_vectorizer()
    X_train_features, X_test_features = extract_features(vectorizer, X_train, X_test)
    
    # 4. Train Model
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_features, y_train)
    
    # 5. Save Model and Vectorizer
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Training complete! Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print("You can now run 'python src/evaluate.py' to test the model.")

if __name__ == "__main__":
    main()
