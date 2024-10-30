import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and the TfidfVectorizer
with open("random_forest_model.pkl", "rb") as model_file:
    model_rf = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit app
def main():
    st.title("Email Spam Classifier")
    st.write("This app uses a machine learning model to classify emails as spam or not spam.")

    # Input text box for email content
    email_text = st.text_area("Enter the email text here:")

    if st.button("Classify Email"):
        if email_text:
            # Transform the input text using the loaded TfidfVectorizer
            email_features = tfidf_vectorizer.transform([email_text])

            # Predict using the loaded Random Forest model
            prediction = model_rf.predict(email_features)

            # Display the result
            if prediction[0] == 1:
                st.write("This email is classified as **Spam**.")
            else:
                st.write("This email is classified as **Not Spam**.")
        else:
            st.write("Please enter some text to classify.")

if __name__ == "__main__":
    main()
