import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (only first time)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- TEXT PREPROCESSING ----------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SMS/EMAIL Spam Detector",
    page_icon="📩",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CSS ----------------
st.markdown("""
<style>

.main{
    background:#f8f9fa;
}

.title{
    text-align:center;
    font-size:48px;
    color:#0E76FD;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    background:#0E76FD;
    color:white;
    height:55px;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#084298;
    color:white;
}

.result{
    padding:18px;
    border-radius:12px;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("📩 SMS/EMAIL Spam Detector")

    st.markdown("---")

    st.write("""
### About

This application predicts whether a message is:

✅ **Not Spam**

or

🚨 **Spam**

using a **Voting Classifier Ensemble Model**.

""")

    st.markdown("---")

    st.success("Model : Voting Classifier")

    st.info("""
Algorithms Used

• SVM

• Multinomial Naive Bayes

• Extra Trees

(Soft Voting)
""")

# ---------------- HEADER ----------------

st.markdown(
"""
<div class="title">
📩 AI SMS Spam Detector
</div>
""",
unsafe_allow_html=True)

st.markdown(
"""
<div class="subtitle">
Machine Learning Project using TF-IDF + Voting Classifier
</div>
""",
unsafe_allow_html=True)

# ---------------- INPUT ----------------

message = st.text_area(
"Enter your Message",
height=200,
placeholder="Type your SMS or Email here..."
)

# ---------------- PREDICT ----------------

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        processed_message = transform_text(message)

        # Convert to dense because model was trained on dense data
        vector = tfidf.transform([processed_message]).toarray()

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        if prediction == 1:

            st.markdown("""
            <div class="result"
            style="background:#ffd6d6;color:#b30000;">
            🚨 SPAM MESSAGE
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability[1]))

            st.metric(
                "Spam Probability",
                f"{probability[1]*100:.2f}%"
            )

        else:

            st.markdown("""
            <div class="result"
            style="background:#d4f8d4;color:#006400;">
            ✅ NOT SPAM
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability[0]))

            st.metric(
                "Not Spam Probability",
                f"{probability[0]*100:.2f}%"
            )

        # Optional Debug Section
        #with st.expander("Processed Text"):
          #  st.write(processed_message)

# ---------------- FOOTER ----------------

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "Voting Classifier")
col2.metric("Vectorizer", "TF-IDF")
col3.metric("Features", "3000")

st.markdown("---")

st.caption("🚀 Built with Streamlit | Voting Classifier | Scikit-learn")