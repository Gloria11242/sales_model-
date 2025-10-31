import streamlit as st
import joblib
import numpy as np

# ==============================
# Page Configuration (must be first Streamlit command)
# ==============================
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="💰",
    layout="centered"
)

# ==============================
# Load Model (with caching)
# ==============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("sales_prediction_model.pkl")
        st.sidebar.success("✅ Model loaded successfully!")  # Optional: Confirms in sidebar
        return model
    except FileNotFoundError:
        st.error(" Model file 'sales_prediction_model.pkl' not found. Upload it to your repo!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Custom CSS to adjust image height
st.markdown(
    """
    <style>
    .banner-img img {
        height: 250px;
        width: 100%;
        max-width: 1600px;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header image (fixed markdown; URL may need checking—see note below)
st.markdown(
    '<div class="banner-img">'
    '<img src="https://www.wavetec.com/wp-content/uploads/2024/01/streamlining-retail-transactions-through-POS-payments.jpg">'
    '</div>',
    unsafe_allow_html=True
)

# ==============================
# Sidebar: Info & Instructions
# ==============================
st.sidebar.title(" About This App")
st.sidebar.markdown(
    """
    ###  Sales Prediction App
    This app predicts **total sales** based on two key inputs:
    - **Quantity Ordered**
    - **Price Each**

    ---
    ###  How the Model Works
    The model was trained on historical sales data to learn patterns between:
    - Number of items sold  
    - Price per item  
    - Total sales revenue  

    When you enter new values, the model estimates the likely sales amount using this relationship:  
    `Predicted Sales ≈ Quantity × Price × (Model Adjustment)`

    ---
    ###  How to Use
    1. Enter the **quantity ordered** and **price per item**.  
    2. Click **Predict Sales**.  
    3. View your predicted total sales instantly.  
    ---
    """
)

# ==============================
# Main App Content
# ==============================
st.title(" Sales Prediction App")
st.subheader("Predict your sales using quantity and price")

st.markdown("###  Enter Your Data")

# Input fields (defaults set to 77 and 89)
quantity = st.number_input("Enter Quantity Ordered:", min_value=0, value=77)
price = st.number_input("Enter Price Each ($):", min_value=0.0, value=89.0, step=0.5)

# ==============================
# Prediction
# ==============================
if st.button(" Predict Sales"):
    X_new = np.array([[quantity, price]])
    try:
        predicted_sales = model.predict(X_new)[0]
        st.success(f" **Predicted Sales:** ${predicted_sales:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
