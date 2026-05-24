# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def main():
    st.title("🛡️ Phishing Email Detection Model")
    st.markdown("""
    This application uses a Machine Learning model (Multinomial Naive Bayes) 
    to classify emails as either **Safe** or **Phishing**. 
    
    Simply paste the content of an email below and click 'Analyze'.
    """)
    
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("Model files not found! Please train the model first by running `python src/train.py`.")
        return
        
    st.subheader("Analyze Email Content")
    email_text = st.text_area("Paste email here:", height=200, placeholder="Dear Customer, your account has been locked...")
    
    if st.button("Analyze Email", type="primary"):
        if email_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Transform the text
                features = vectorizer.transform([email_text])
                
                # Predict
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Display Results
                st.markdown("---")
                st.subheader("Analysis Result")
                
                # 1 = Phishing, 0 = Safe
                if prediction == 1:
                    st.error("🚨 **WARNING: This email is classified as PHISHING!**")
                    st.progress(float(probabilities[1]))
                    st.caption(f"Confidence: {probabilities[1]*100:.2f}%")
                else:
                    st.success("✅ **SAFE: This email appears to be legitimate.**")
                    st.progress(float(probabilities[0]))
                    st.caption(f"Confidence: {probabilities[0]*100:.2f}%")
                
                with st.expander("Why did it make this prediction?"):
                    st.write("""
                    The model analyzes the text using TF-IDF (Term Frequency-Inverse Document Frequency) 
                    and looks for patterns, keywords, and structural anomalies common in phishing emails, 
                    such as urgency ("immediate action required") or suspicious links.
                    """)

if __name__ == "__main__":
    main()
