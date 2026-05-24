import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Loads the email dataset from a CSV file.
    Expects columns 'text' (email content) and 'label' ('Safe' or 'Phishing').
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets.
    Converts labels to binary: 'Phishing' -> 1, 'Safe' -> 0.
    """
    # Convert labels to binary format
    # Assuming the labels are exactly "Phishing" and "Safe"
    df['label_num'] = df['label'].map({'Phishing': 1, 'Safe': 0})
    
    # Drop rows with NaN labels if any mapping failed
    df = df.dropna(subset=['label_num'])
    
    X = df['text']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test
