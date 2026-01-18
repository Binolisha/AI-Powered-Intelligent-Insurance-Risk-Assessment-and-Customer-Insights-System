import streamlit as st
import pickle
import pandas as pd
import json
import re
import os
import numpy as np
import torch
from collections import Counter
from fuzzywuzzy import process
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.models import load_model,Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import docx

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(
    page_title="Insurance AI Platform",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Insurance AI Platform")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Risk Classification",
        "Claim Prediction",
        "Customer Segmentation",
        "Anomaly Detection",
        "Dimensionality Reduction",
        "Document Translation",
        "Sentiment Analysis",
        "Policy Summarization",
        "Insurance Chatbot"
    ]
)

# =====================================================
# HOME
# =====================================================
if page == "Home":
    st.subheader("Welcome")
    st.markdown("""
    **End-to-End AI Solutions for the Insurance Industry**

    ✔ Risk Assessment  
    ✔ Claim Prediction  
    ✔ Fraud Detection  
    ✔ Customer Segmentation  
    ✔ NLP-powered Translation, Sentiment & Summarization  
    ✔ Automated Customer Support
    """)

# =====================================================
# 1. RISK CLASSIFICATION
# =====================================================
elif page == "Risk Classification":
    st.header("📊 Insurance Risk Classification")

    @st.cache_resource
    def load_model():
        model_path = r"C:\Users\DELL\Downloads\insurance_risk_classification_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file NOT found at:\n{model_path}")
            st.stop()
        with open(model_path, "rb") as f:
            return pickle.load(f)

    model = load_model()

    # ==============================
    # INPUTS
    # ==============================

    claim_amount_SZL = st.number_input("Enter the Claim Amount", min_value=0.0, step=0.1)
    poverty_index = st.slider("Enter the Poverty Index", 0, 60)
    age = st.number_input("Enter your age", 18, 100)
    financial_stability_index = st.slider("Enter the Financial Stability Index", 0, 100)
    dependents_count = st.number_input("Enter number of Dependents", 0, 5)

    # ------------------------------
    # GENDER
    # ------------------------------
    gender = 1 if st.selectbox("Gender", ["Male", "Female"]) == "Female" else 0

    # ------------------------------
    # YES / NO FIELDS
    # ------------------------------
    def yes_no(label):
        return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

    multiple_claims_flag = yes_no("Multiple Claims")
    late_payment_history = yes_no("Late Payment History")
    rural_vs_urban = 1 if st.selectbox("Location", ["Rural", "Urban"]) == "Urban" else 0
    policy_lapse_history = yes_no("Policy Lapse History")

    # ------------------------------
    # CLAIM TYPE (ONE-HOT)
    # ------------------------------
    claim_type = st.selectbox(
        "Claim Type",
        [
            "fire damage",
            "health emergency",
            "livestock loss",
            "road accident",
            "storm damage",
            "theft",
        ],
    )

    claim_type_fire_damage = int(claim_type == "fire damage")
    claim_type_health_emergency = int(claim_type == "health emergency")
    claim_type_livestock_loss = int(claim_type == "livestock loss")
    claim_type_road_accident = int(claim_type == "road accident")
    claim_type_storm_damage = int(claim_type == "storm damage")
    claim_type_theft = int(claim_type == "theft")

    # ------------------------------
    # POLICY TYPE (ONE-HOT)
    # ------------------------------
    policy_type = st.selectbox(
        "Policy Type",
        [
            "funeral cover",
            "health insurance",
            "home insurance",
            "life insurance",
            "vehicle insurance",
        ],
    )

    policy_type_funeral_cover = int(policy_type == "funeral cover")
    policy_type_health_insurance = int(policy_type == "health insurance")
    policy_type_home_insurance = int(policy_type == "home insurance")
    policy_type_life_insurance = int(policy_type == "life insurance")
    policy_type_vehicle_insurance = int(policy_type == "vehicle insurance")

    # ==============================
    # MODEL INPUTS (EXACT ORDER)
    # ==============================
    inputs = {
        "claim_amount_SZL": claim_amount_SZL,
        "poverty_index": poverty_index,
        "age": age,
        "financial_stability_index": financial_stability_index,
        "dependents_count": dependents_count,
        "gender": gender,
        "multiple_claims_flag": multiple_claims_flag,
        "late_payment_history": late_payment_history,
        "rural_vs_urban": rural_vs_urban,
        "policy_lapse_history": policy_lapse_history,

        "claim_type_fire damage": claim_type_fire_damage,
        "claim_type_health emergency": claim_type_health_emergency,
        "claim_type_livestock loss": claim_type_livestock_loss,
        "claim_type_road accident": claim_type_road_accident,
        "claim_type_storm damage": claim_type_storm_damage,
        "claim_type_theft": claim_type_theft,

        "policy_type_funeral cover": policy_type_funeral_cover,
        "policy_type_health insurance": policy_type_health_insurance,
        "policy_type_home insurance": policy_type_home_insurance,
        "policy_type_life insurance": policy_type_life_insurance,
        "policy_type_vehicle insurance": policy_type_vehicle_insurance,
    }

    df = pd.DataFrame([inputs])

    # 🔒 Force exact training column order
    df = df[model.get_booster().feature_names]

    if st.button("Predict Risk"):
        prediction = model.predict(df)
        st.success(f"Predicted Risk Class: **{prediction[0]}**")

# =====================================================
# 2. CLAIM PREDICTION
# =====================================================
elif page == "Claim Prediction":
    st.header("💰 Claim Amount Prediction")

    import os, pickle, pandas as pd

    @st.cache_resource
    def load_model():
        model_path = r"C:\Users\DELL\Downloads\svr_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file NOT found at:\n{model_path}")
            st.stop()
        with open(model_path, "rb") as f:
            return pickle.load(f)

    model = load_model()

    feature_names = [
        "age","income","value_of_home","travel_time","vehicle_age",
        "5_year_num_of_claims","num_young_drivers","highest_education",
        "single_parent","married","gender","type_of_use","licence_revoked",
        "address_type",
        "occupation_Clerical","occupation_Doctor","occupation_Home Maker",
        "occupation_Lawyer","occupation_Manager","occupation_Professional",
        "occupation_Student",
        "vehicle_type_Panel Truck","vehicle_type_Pickup",
        "vehicle_type_SUV","vehicle_type_Sports Car","vehicle_type_Van"
    ]

    inputs = {}

    # ---------- NUMERICAL ----------
    st.subheader("Numerical Details")
    inputs["age"] = st.number_input("Age", min_value=0)
    inputs["income"] = st.number_input("Income", min_value=0.0)
    inputs["value_of_home"] = st.number_input("Value of Home", min_value=0.0)
    inputs["travel_time"] = st.number_input("Travel Time", min_value=0.0)
    inputs["vehicle_age"] = st.number_input("Vehicle Age", min_value=0)
    inputs["5_year_num_of_claims"] = st.number_input("Claims in Last 5 Years", min_value=0)
    inputs["num_young_drivers"] = st.number_input("Number of Young Drivers", min_value=0)

    # ---------- ORDINAL ----------
    st.subheader("Education")
    education = st.selectbox(
        "Highest Education",
        ["< High School", "High School", "Bachelors", "Masters", "PhD"]
    )
    inputs["highest_education"] = {
        "< High School": 0,
        "High School": 1,
        "Bachelors": 2,
        "Masters": 3,
        "PhD": 4
    }[education]

    # ---------- BINARY ----------
    st.subheader("Personal Information")
    inputs["single_parent"] = 1 if st.radio("Single Parent", ["No", "Yes"]) == "Yes" else 0
    inputs["married"] = 1 if st.radio("Married", ["No", "Yes"]) == "Yes" else 0
    inputs["gender"] = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0
    inputs["licence_revoked"] = 1 if st.radio("Licence Revoked", ["No", "Yes"]) == "Yes" else 0

    # ---------- CATEGORICAL ----------
    st.subheader("Usage & Address")
    inputs["type_of_use"] = 1 if st.selectbox("Type of Use", ["Private", "Commercial"]) == "Commercial" else 0
    inputs["address_type"] = 1 if st.selectbox(
        "Address Type",
        ["Highly Rural / Rural", "Highly Urban / Urban"]
    ) == "Highly Urban / Urban" else 0

    # ---------- ONE-HOT: OCCUPATION (Blue Collar DROPPED) ----------
    st.subheader("Occupation")
    occupation_options = [
        "Clerical", "Doctor", "Home Maker",
        "Lawyer", "Manager", "Professional", "Student"
    ]
    selected_occupation = st.selectbox(
        "Occupation",
        ["Blue Collar"] + occupation_options
    )

    for occ in occupation_options:
        inputs[f"occupation_{occ}"] = 1 if selected_occupation == occ else 0

    # ---------- ONE-HOT: VEHICLE TYPE (Minivan DROPPED) ----------
    st.subheader("Vehicle Type")
    vehicle_types = [
        "Panel Truck", "Pickup", "SUV", "Sports Car", "Van"
    ]
    selected_vehicle = st.selectbox(
        "Vehicle Type",
        ["Minivan"] + vehicle_types
    )

    for vt in vehicle_types:
        inputs[f"vehicle_type_{vt}"] = 1 if selected_vehicle == vt else 0

    # ---------- PREDICTION ----------
    if st.button("Predict Claim"):
        X = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)
        prediction = model.predict(X)
        st.success(f"Estimated Claim Amount: **{prediction[0]:,.2f}**")

# =====================================================
# 3. CUSTOMER SEGMENTATION
# =====================================================
elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")

    import os, pickle, pandas as pd

    @st.cache_resource
    def load_model():
        model_path = r"C:\Users\DELL\Downloads\customersegmentation.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file NOT found at:\n{model_path}")
            st.stop()
        with open(model_path, "rb") as f:
            return pickle.load(f)

    model = load_model()

    features = [
        'BirthYear','EducDeg','Children',
        'MonthSal_binned_(0, 1000]',
        'MonthSal_binned_(1000, 2000]',
        'MonthSal_binned_(2000, 3000]',
        'MonthSal_binned_(3000, 4000]',
        'MonthSal_binned_(4000, 5000]',
        'MonthSal_binned_(>5000]',
        'PremMotor','PremHousehold',
        'PremHealth','PremLife','PremWork','ClaimsRate'
    ]

    inputs = {}

    # ---------------- NUMERICAL ----------------
    st.subheader("Basic & Premium Information")

    inputs["BirthYear"] = st.number_input("Birth Year", min_value=1900, max_value=2025)
    inputs["PremMotor"] = st.number_input("Motor Insurance Premium", min_value=0.0)
    inputs["PremHousehold"] = st.number_input("Household Insurance Premium", min_value=0.0)
    inputs["PremHealth"] = st.number_input("Health Insurance Premium", min_value=0.0)
    inputs["PremLife"] = st.number_input("Life Insurance Premium", min_value=0.0)
    inputs["PremWork"] = st.number_input("Work Insurance Premium", min_value=0.0)
    inputs["ClaimsRate"] = st.number_input("Claims Rate", min_value=0.0)

    # ---------------- EDUCATION (ORDINAL) ----------------
    st.subheader("Education")

    educ = st.selectbox(
        "Educational Degree",
        ["1 - Basic", "2 - High School", "3 - BSc/MSc", "4 - PhD/Other"]
    )

    inputs["EducDeg"] = {
        "1 - Basic": 1,
        "2 - High School": 2,
        "3 - BSc/MSc": 3,
        "4 - PhD/Other": 4
    }[educ]

    # ---------------- CHILDREN (BINARY) ----------------
    st.subheader("Family")

    inputs["Children"] = 1 if st.radio("Has Children?", ["No", "Yes"]) == "Yes" else 0

    # ---------------- SALARY (ONE-HOT) ----------------
    st.subheader("Monthly Salary")

    salary_bins = [
        "(0, 1000]",
        "(1000, 2000]",
        "(2000, 3000]",
        "(3000, 4000]",
        "(4000, 5000]",
        "(>5000]"
    ]

    selected_salary = st.selectbox("Monthly Salary Range", salary_bins)

    for bin_ in salary_bins:
        inputs[f"MonthSal_binned_{bin_}"] = 1 if bin_ == selected_salary else 0

    # ---------------- FINAL DATAFRAME ----------------
    X = pd.DataFrame([[inputs[f] for f in features]], columns=features)

    if st.button("Segment Customer"):
        segment = model.predict(X)
        st.success(f"Customer Segment: **{segment[0]}**")

# =====================================================
# 4. ANOMALY DETECTION
# =====================================================
elif page == "Anomaly Detection":
    st.header("🚨 Fraud & Anomaly Detection")
    
    @st.cache_resource
    def load_model():
        model_path = r"C:\Users\DELL\Downloads\isolation_forest_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file NOT found at:\n{model_path}")
            st.stop()
        with open(model_path, "rb") as f:
            return pickle.load(f)

    model = load_model()

    feature_names = [
        "amount",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
        "is_merchant",
        "is_large",
        "rule_violation"
    ]

    # ---------- NUMERICAL INPUT ----------
    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)

    # ---------- NOMINAL INPUT ----------
    st.subheader("Transaction Type")
    transaction_type = st.selectbox(
        "Select Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    # ---------- BINARY INPUT ----------
    st.subheader("Transaction Flags")
    is_merchant = 1 if st.checkbox("Merchant") else 0
    is_large = 1 if st.checkbox("Large Transaction") else 0
    rule_violation = 1 if st.checkbox("Rule Violation") else 0

    # ---------- ENCODE INPUTS ----------
    inputs = {"amount": amount}

    # One-hot encode transaction type
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        inputs[f"type_{t}"] = 1 if transaction_type == t else 0

    # Binary features
    inputs["is_merchant"] = is_merchant
    inputs["is_large"] = is_large
    inputs["rule_violation"] = rule_violation

    # Convert to DataFrame
    X = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)

    st.subheader("Transaction Input Preview")
    st.dataframe(X)

    # ---------- ANOMALY DETECTION ----------
    if st.button("Detect Anomaly"):
        result = model.predict(X)
        if result[0] == -1:
            st.error("🚩 Anomaly Detected")
        else:
            st.success("✅ Normal Transaction")

# ===============================
# Page: Dimensionality Reduction
# ===============================

elif page == "Dimensionality Reduction":
    st.header("📉 Autoencoder Dimensionality Reduction")

    # -------------------------------
    # Load encoder (cached)
    # -------------------------------
    @st.cache_resource
    def load_encoder():
        model_path = r"C:\Users\DELL\Downloads\autoencoder_model.h5"

        # Load autoencoder for inference only
        autoencoder = load_model(model_path, compile=False)

        # ⚠️ Check if latent layer exists
        layer_names = [layer.name for layer in autoencoder.layers]
        if "latent_space" not in layer_names:
            st.error("Latent layer 'latent_space' not found in the model!")
            st.stop()

        # ⚠️ Assumes latent layer is layer index 1
        encoder = Model(
            inputs=autoencoder.input,
            outputs=autoencoder.get_layer("latent_space").output
        )

        return encoder

    encoder = load_encoder()

    st.markdown(
        """
        **Objective:**  
        Reduce high-dimensional customer data into a compact latent representation  
        for improved model performance and interpretability.
        """
    )

    # -------------------------------
    # User input
    # -------------------------------
    data = st.text_area(
        "Paste numerical feature values (comma-separated)",
        placeholder="Example: 28, 18000, 1, 60000, 5, 3, 2, 1, 1, 1, 1, 1"
    )

    # -------------------------------
    # Run dimensionality reduction
    # -------------------------------
    if st.button("Reduce Dimensions"):
        if not data.strip():
            st.error("❌ Input cannot be empty.")
            st.stop()

        try:
            # 🔹 Remove hidden unicode characters
            cleaned = re.sub(r"[^\d\.\-\,]", "", data)

            # 🔹 Convert to float safely
            values = [float(x) for x in cleaned.split(",") if x.strip() != ""]

            X = np.array(values, dtype=np.float32).reshape(1, -1)

            # -------------------------------
            # Feature count validation
            # -------------------------------
            expected_features = encoder.input_shape[1]

            if X.shape[1] != expected_features:
                st.error(
                    f"❌ Feature mismatch: Model expects **{expected_features}** features "
                    f"but received **{X.shape[1]}**."
                )
                st.stop()

            # -------------------------------
            # Encode input
            # -------------------------------
            encoded = encoder.predict(X, verbose=0)

            # -------------------------------
            # Output
            # -------------------------------
            st.success("Dimensionality Reduction Successful ✅")

            st.subheader("Original Input")
            st.write(X)
            st.write(f"Original Dimensions: **{X.shape[1]}**")

            st.subheader("Encoded Representation (Latent Space)")
            st.write(encoded)
            st.write(f"Reduced Dimensions: **{encoded.shape[1]}**")

        except ValueError:
            st.error("❌ Please enter only numeric values separated by commas.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            
# =====================================================
# 6. TEXT TRANSLATION
# =====================================================
elif page == "Document Translation":
    st.header("🌍 Insurance Text Translation")

    MODEL_PATH = r"D:\models\nllb_insurance_translator"

    # ---------------- LOAD MODEL ONCE ----------------
    @st.cache_resource
    def load_translation_model():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        return tokenizer, model

    tokenizer, model = load_translation_model()

    # ---------------- TEXT INPUT ----------------
    st.subheader("✍️ Enter Insurance Text")
    input_text = st.text_area(
        "Paste or type the text you want to translate",
        height=200
    )

    # ---------------- LANGUAGE SELECTION ----------------
    st.subheader("🌐 Target Language")

    language_map = {
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda"
    }


    target_language = st.selectbox(
        "Select Target Language",
        list(language_map.keys())
    )

    # ---------------- TRANSLATION ----------------
    if st.button("Translate Text"):
        if input_text.strip() == "":
            st.warning("⚠️ Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                tokenizer.src_lang = "eng_Latn"

                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                forced_bos_token_id = tokenizer.convert_tokens_to_ids(
                    language_map[target_language])
                
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512
                )

                translated_text = tokenizer.decode(
                    translated_tokens[0],
                    skip_special_tokens=True
                )

            st.success("✅ Translation Completed")
            st.subheader("📝 Translated Text")
            st.text_area("Output", translated_text, height=250)

# =====================================================
# 7. SENTIMENT ANALYSIS
# =====================================================
elif page == "Sentiment Analysis":
    st.header("🧠 Sentiment Analysis")
    @st.cache_resource
    def load_artifacts():
        model_path = r"C:\Users\DELL\Downloads\model_sentimental_analysis.pkl"
        vectorizer_path = r"C:\Users\DELL\Downloads\tfidfvectorizer.pkl"

        if not os.path.exists(model_path):
            st.error(f"❌ Model file NOT found at:\n{model_path}")
            st.stop()

        if not os.path.exists(vectorizer_path):
            st.error(f"❌ TF-IDF Vectorizer not found:\n{vectorizer_path}")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    model, vectorizer = load_artifacts()

    # ---------------- TEXT CLEANING ----------------
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    # ---------------- INPUT ----------------
    feedback = st.text_area("Customer Feedback", height=150)

    # ---------------- PREDICTION ----------------
    if st.button("Analyze Sentiment"):
        if feedback.strip() == "":
            st.warning("⚠️ Please enter some feedback text.")
        else:
            cleaned_text = clean_text(feedback)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            # Label mapping
            sentiment_map = {1: "😊 Positive",0: "😠 Negative",2: "😐 Neutral"}

            sentiment_label = sentiment_map.get(prediction[0], "Unknown")

            st.success(f"Sentiment: **{sentiment_label}**")


# =====================================================
# 8. POLICY SUMMARIZATION
# =====================================================
elif page == "Policy Summarization":
    st.header("📄 Policy Summarization")
    # -----------------------------
    # Text Cleaning Function
    # -----------------------------
    def clean_text(text):
        text = re.sub(r'\n+', ' ', text)      # remove newlines
        text = re.sub(r'\s+', ' ', text)      # remove extra spaces
        return text.strip()

    # -----------------------------
    # Extractive Summary Function
    # -----------------------------
    def extractive_summary(text, num_sentences=3):
        if not text.strip():
            return "❌ Please enter some text to summarize."
    
        text = clean_text(text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_freq = Counter(words)

        # Score sentences
        sentence_scores = {}
        for sent in sentences:
            sent_words = word_tokenize(sent.lower())
            score = sum(word_freq.get(word, 0) for word in sent_words)
            sentence_scores[sent] = score

        # Select top N sentences
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        return " ".join(top_sentences)

    text = st.text_area("Paste Insurance Policy",placeholder="Paste your insurance policy text here...")

    num_sentences = st.number_input(
        "Number of sentences for summary",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    if st.button("Summarize"):
        summary = extractive_summary(text, num_sentences)
        st.success(summary)

# =====================================================
# 9. INSURANCE CHATBOT
# =====================================================
elif page == "Insurance Chatbot":
    st.header("🤖 Insurance FAQ Chatbot")

    with open(r"C:\Users\DELL\Downloads\insurance_faq_100.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    questions = [q["question"] for q in faq_data]
    user_input = st.text_input("Ask a question")

    if st.button("Get Answer"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            match, score = process.extractOne(user_input, questions)
            if score >= 60:
                answer = next(q["answer"] for q in faq_data if q["question"] == match)
                st.success(answer)
            else:
                st.error("❌ Sorry, no matching FAQ found.")