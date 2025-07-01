import streamlit as st
import pickle

# ✅ Ensure set_page_config is at the top
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

# ✅ Try to load the vectorizer and model
try:
    with open("new_tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open("new_emotion_classifier.pkl", "rb") as file:
        model = pickle.load(file)

    st.success("✅ Model and vectorizer loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()  # 🔹 Stop execution if loading fails

# Function to predict emotion
def predict_emotion(text):
    transformed_text = vectorizer.transform([text])  # Transform text using TF-IDF
    prediction = model.predict(transformed_text)  # Predict emotion
    return prediction[0]

# Streamlit UI
st.title("🎭 Emotion Detection System")
st.write("💬 Enter a text message to detect its emotion.")

# Text input
user_input = st.text_area("📝 Enter your text below:")

if st.button("🔍 Detect Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"🎭 **Detected Emotion:** {emotion}")
    else:
        st.warning("⚠️ Please enter some text.")

st.markdown("---")
st.markdown("💡 *Powered by Machine Learning & NLP*")
