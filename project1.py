import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

#PAGE CONFIG 
st.set_page_config(
    page_title="Food Review Sentiment AI",
    page_icon="🍽️",
    layout="wide")

#CUSTOM CSS 
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#FF4B4B;
        text-align:center;
    }
    .sub-text {
        text-align:center;
        color:gray;
        font-size:18px;
    }
    </style>
""", unsafe_allow_html=True)



#TITLE 
st.markdown('<p class="main-title">🍽️ Food Sentiment Analysis AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Analyze customer food reviews using Machine Learning</p>', unsafe_allow_html=True)

#CLEANING FUNCTION
def mycleaning(doc):
    if not doc:
        return ""
    return re.sub("[^a-zA-Z ]", "", doc).lower()

st.markdown(
    """
    <style>
    .stApp {
        background-image: 
        linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
        url("https://images.unsplash.com/photo-1504674900247-0877df9cc836");
        
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Main text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    /* Sidebar FIX */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }

    /* Input box */
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True)


#LOAD MODEL
model = joblib.load("Sentiment_model.pkl")

#SIDEBAR
st.sidebar.header("🎛️ Control Panel")
st.sidebar.image("TOO_restaurant_.jpg")
st.sidebar.header("📌 Project Info")
st.sidebar.write("This AI model predicts whether a food review is Positive or Negative.")

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Libraries Used")

st.sidebar.write("""
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Regex (re)  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📬 Contact Information")

st.sidebar.markdown("""
📧 **Email:** bhuvank0412@gmail.com  
🔗 **LinkedIn:** https://www.linkedin.com/in/bhuvan-kumar-72897a376  
📞 **Phone:** 99584XXXX  
""")


st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit")

#SINGLE PREDICTION
st.markdown("## ✍️ Enter Your Review")
sample = st.text_input("Type your food review here...")

col1, col2 = st.columns([1, 5])

with col1:
    predict_btn = st.button("🚀 Predict")

if predict_btn:
    if sample.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned = mycleaning(sample)

        try:
            pred = model.predict([cleaned])
            prob = model.predict_proba([cleaned])

            st.markdown("### 📊 Result")

            if pred[0] == 0:
                st.error("😞 Negative Review")
                st.progress(int(prob[0][0] * 100))
                st.write(f"Confidence: {prob[0][0]:.2f}")
            else:
                st.success("😊 Positive Review")
                st.progress(int(prob[0][1] * 100))
                st.write(f"Confidence: {prob[0][1]:.2f}")
                st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

#BULK PREDICTION
st.markdown("---")
st.markdown("## 📂 Bulk Review Analysis")

file = st.file_uploader("Upload CSV/TXT file with reviews", type=["csv", "txt"])

if file:
    try:
        df = pd.read_csv(file, names=["Review"])
        st.write("### 📄 Uploaded Data")
        placeholder = st.empty()
        placeholder.dataframe(df)

        if st.button("⚡ Run Bulk Prediction"):
            corpus = df["Review"].apply(mycleaning)

            pred = model.predict(corpus)
            prob = np.max(model.predict_proba(corpus), axis=1)

            df["Sentiment"] = pred
            df["Confidence"] = prob
            df["Sentiment"] = df["Sentiment"].map({0: 'Dislike 😞', 1: 'Like 😊'})

            st.write("### ✅ Prediction Results")
            placeholder.dataframe(df)

    except Exception as e:
        st.error(f"File Error: {e}")

#FOOTER
st.markdown("---")
st.caption("🚀 Built with Streamlit | Machine Learning Project | By You")
