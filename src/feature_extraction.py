from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer(max_features=5000):
    """
    Initializes and returns a TF-IDF Vectorizer.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) will convert
    the text documents into a matrix of TF-IDF features. This helps
    in emphasizing important words and de-emphasizing common words.
    """
    # We use stop_words='english' to remove common words like 'the', 'is', etc.
    # lowercase=True ensures 'Click' and 'click' are treated the same
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=max_features
    )
    return vectorizer

def extract_features(vectorizer, X_train, X_test=None):
    """
    Fits the vectorizer on the training data and transforms both training and testing data.
    """
    print("Extracting features using TF-IDF...")
    X_train_features = vectorizer.fit_transform(X_train)
    
    if X_test is not None:
        X_test_features = vectorizer.transform(X_test)
        return X_train_features, X_test_features
    
    return X_train_features
