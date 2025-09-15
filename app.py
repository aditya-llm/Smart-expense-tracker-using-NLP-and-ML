import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import joblib
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Prediction", "📊 Dashboard", "🛠 Corrections"])

# -----------------------------
# Dashboard
# -----------------------------
def show_dashboard():
    st.header("📊 Expense Dashboard")
    if not os.path.exists("predicted_expenses.csv"):
        st.info("No expense logs yet. Start by entering some expenses!")
        return

    df = pd.read_csv("predicted_expenses.csv")

    # Drop empty categories if exist
    df = df.dropna(subset=['predicted_category'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Metrics
    total_spent = df['total_amount'].sum(skipna=True)
    total_logs = len(df)
    avg_spent = df['total_amount'].mean(skipna=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Spent", f"₹{total_spent:,.2f}")
    col2.metric("📝 Total Logs", total_logs)
    col3.metric("📉 Avg Expense", f"₹{avg_spent:,.2f}")

    st.markdown("---")

    # Expenses by category
    st.subheader("📂 Expenses by Category")
    cat_expense = df.groupby("predicted_category")['total_amount'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=cat_expense.values, y=cat_expense.index, ax=ax)
    ax.set_xlabel("Total Expense (₹)")
    ax.set_ylabel("Category")
    st.pyplot(fig)

    st.markdown("---")

    # Expenses over time
    st.subheader("📅 Expenses Over Time")
    df['date'] = df['timestamp'].dt.date
    daily_expense = df.groupby("date")['total_amount'].sum()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    daily_expense.plot(ax=ax2, marker="o")
    ax2.set_ylabel("Total Expense (₹)")
    st.pyplot(fig2)

    st.markdown("---")

    # Show table
    st.subheader("📜 Expense Logs")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(20), use_container_width=True)

# -----------------------------
# Setup & Downloads
# -----------------------------
@st.cache_resource
def setup_nltk():
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = setup_nltk()

# -----------------------------
# Text Cleaning with NLTK
# -----------------------------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'₹|\$|,|\.', '', s)
    s = re.sub(r'\d{3,}', '<AMOUNT>', s)
    
    tokens = word_tokenize(s)
    tokens = [token for token in tokens if re.match(r'^[a-z0-9\+\-\@\_\&<>\s]+$', token)]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    cleaned = ' '.join(tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# -----------------------------
# Extract Amounts
# -----------------------------
def extract_amounts(text):
    if not isinstance(text, str):
        return []
    matches = re.findall(r'(\d+\.?\d*)', text)
    return [float(match) for match in matches]

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/expense_pipeline.pkl"
TRAIN_FILE = "expense.csv"
CORRECTIONS_FILE = "corrections.csv"
PREDICTED_FILE = "predicted_expenses.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please train it first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_category(text):
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned]).max()
    return pred, proba

# -----------------------------
# Retrain Model (with corrections)
# -----------------------------
def retrain_model():
    if not os.path.exists(TRAIN_FILE):
        st.warning("Training data not found. Skipping retrain.")
        return

    df_train = pd.read_csv(TRAIN_FILE)
    df_train = df_train[df_train['Category'] != "Category"]
    df_train = df_train.dropna(subset=['Category'])
    df_train['Category'] = df_train['Category'].str.strip()

    if os.path.exists(CORRECTIONS_FILE):
        df_corr = pd.read_csv(CORRECTIONS_FILE)
        df_corr = df_corr.dropna(subset=['correct_category'])
        df_corr = df_corr.rename(columns={'text': 'Notes', 'correct_category': 'Category'})
        df_corr['For What'] = ""
        df_train = pd.concat([df_train, df_corr[['Notes', 'For What', 'Category']]], ignore_index=True)
        st.info(f"🔁 Added {len(df_corr)} corrections to training data.")

    df_train['text'] = (df_train['Notes'].fillna('') + ' ' + df_train['For What'].fillna('')).map(clean_text)
    X = df_train['text']
    y = df_train['Category']

    pipeline_new = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=5000)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='ovr'))
    ])

    pipeline_new.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline_new, MODEL_PATH)
    st.success(f"✅ Model retrained and saved to {MODEL_PATH}")

# -----------------------------
# Log Expense
# -----------------------------
def log_expense(expense_text):
    predicted_cat, confidence = predict_category(expense_text)
    cleaned_text = clean_text(expense_text)
    amounts = extract_amounts(expense_text)
    total_amount = sum(amounts) if amounts else None

    row = {
        'original_text': expense_text,
        'cleaned_text': cleaned_text,
        'predicted_category': predicted_cat,
        'confidence': round(confidence, 4),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'amounts': str(amounts),
        'total_amount': total_amount,
        'corrected': None
    }

    # --- ACTIVE LEARNING ---
    if confidence >= 0.6:
        row['corrected'] = 0  # auto accepted
    else:
        row['corrected'] = None  # flag for manual correction

    # Save to predicted_expenses.csv
    df_new = pd.DataFrame([row])
    if os.path.exists(PREDICTED_FILE):
        df_existing = pd.read_csv(PREDICTED_FILE)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(PREDICTED_FILE, index=False)

    return row, confidence < 0.6  # return if needs correction

# -----------------------------
# Save Correction
# -----------------------------
def save_correction(expense_text, correct_category):
    corr_row = {
        'text': expense_text,
        'correct_category': correct_category,
        'source': 'user_correction',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_corr = pd.DataFrame([corr_row])
    if os.path.exists(CORRECTIONS_FILE):
        df_corr.to_csv(CORRECTIONS_FILE, mode='a', header=False, index=False)
    else:
        df_corr.to_csv(CORRECTIONS_FILE, index=False)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Expense Tracker", layout="centered")

st.title("🤖 AI Expense Tracker with Active Learning")
st.markdown("Enter an expense description, and the model will predict its category. Low-confidence predictions can be corrected — these help improve the model over time!")

# Auto-retrain on startup if corrections exist
if os.path.exists(CORRECTIONS_FILE) and os.path.getsize(CORRECTIONS_FILE) > 0:
    with st.spinner("🔍 Found untrained corrections. Retraining model..."):
        retrain_model()
        # Clear corrections after retrain? Optional — you commented it out to avoid header deletion
        # But if you want to clear, do it safely:
        # if os.path.exists(CORRECTIONS_FILE):
        #     pd.DataFrame(columns=['text','correct_category','source','timestamp']).to_csv(CORRECTIONS_FILE, index=False)

# Input form
with st.form("expense_form"):
    user_input = st.text_input("📝 Enter expense description (e.g., '30 for coffee')", "")
    submitted = st.form_submit_button("Predict & Log")

if submitted and user_input.strip():
    with st.spinner("Predicting..."):
        row, needs_correction = log_expense(user_input)

    st.subheader("📊 Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Category", row['predicted_category'])
    col2.metric("Confidence", f"{row['confidence']:.2%}")
    col3.metric("Total Amount", f"₹{row['total_amount']}" if row['total_amount'] else "N/A")

    if needs_correction:
        st.warning("⚠️ Low confidence prediction. Please correct the category below.")
        correct_cat = st.text_input("✏️ Enter correct category", key="correction_input")
        if st.button("✅ Confirm Correction", key="confirm_btn"):
            if correct_cat.strip():
                save_correction(user_input, correct_cat.strip())
                # Also update the logged row's 'corrected' field
                df = pd.read_csv(PREDICTED_FILE)
                df.loc[df['original_text'] == user_input, 'corrected'] = correct_cat.strip()
                df.to_csv(PREDICTED_FILE, index=False)
                st.success(f"✔️ Saved correction: '{correct_cat}' for '{user_input}'")
                st.balloons()
            else:
                st.error("Please enter a valid category.")

    else:
        st.success("✅ Auto-accepted with high confidence.")

# Show recent logs
st.subheader("📜 Recent Predictions")
if os.path.exists(PREDICTED_FILE):
    df_logs = pd.read_csv(PREDICTED_FILE)
    st.dataframe(df_logs.tail(10).sort_values('timestamp', ascending=False), use_container_width=True)
else:
    st.info("No logs yet. Start entering expenses!")

# Optional: Retrain button
if st.button("🔄 Retrain Model Now (using all corrections)"):
    with st.spinner("Retraining..."):
        retrain_model()
    st.success("Model retrained successfully!")


# -----------------------------
# Corrections Viewer
# -----------------------------
def show_corrections():
    st.header("🛠 User Corrections")
    if not os.path.exists("corrections.csv"):
        st.info("No corrections have been logged yet.")
        return

    df_corr = pd.read_csv("corrections.csv")
    st.dataframe(df_corr.sort_values("timestamp", ascending=False), use_container_width=True)


# -----------------------------
# Router
# -----------------------------
if page == "🔮 Prediction":
    st.title("🤖 AI Expense Tracker with Active Learning")
    st.markdown("Enter an expense description, and the model will predict its category. Low-confidence predictions can be corrected — these help improve the model over time!")
elif page == "📊 Dashboard":
    show_dashboard()
elif page == "🛠 Corrections":
    show_corrections()
