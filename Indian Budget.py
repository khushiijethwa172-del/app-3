import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Budget Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

.title{
    text-align:center;
    color:#0E4D92;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.card{
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0px 0px 10px rgba(0,0,0,.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("<h1 class='title'>💰 AI Budget Forecasting System</h1>", unsafe_allow_html=True)

st.markdown(
"<p class='subtitle'>Predict Next Year's Budget using Machine Learning</p>",
unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Upload Budget",
        "Data Preview",
        "Prediction",
        "Dashboard"
    ]
)

# ---------------- PAGE 1 ----------------

if page=="Upload Budget":

    st.header("📂 Upload Previous Budget Files")

    uploaded_files = st.file_uploader(

        "Upload Budget Excel/CSV Files",

        type=["csv","xlsx"],

        accept_multiple_files=True

    )

    if uploaded_files:

        os.makedirs("uploads",exist_ok=True)

        for file in uploaded_files:

            save_path=os.path.join("uploads",file.name)

            with open(save_path,"wb") as f:

                f.write(file.getbuffer())

        st.success("Files Uploaded Successfully ✅")

        st.balloons()

# ---------------- PAGE 2 ----------------

elif page=="Data Preview":

    st.header("📊 Uploaded Budget Files")

    folder="uploads"

    if not os.path.exists(folder):

        st.warning("No Files Uploaded")

    else:

        files=os.listdir(folder)

        if len(files)==0:

            st.warning("Folder Empty")

        else:

            selected=st.selectbox("Choose File",files)

            path=os.path.join(folder,selected)

            if selected.endswith(".csv"):

                df=pd.read_csv(path)

            else:

                df=pd.read_excel(path)

            st.dataframe(df,use_container_width=True)

# ---------------- PAGE 3 ----------------

elif page=="Prediction":

    st.header("🤖 Budget Prediction")

    year=st.number_input(
        "Predict Budget For Year",
        2025,
        2050,
        2026
    )

    inflation=st.slider(
        "Inflation %",
        0,
        20,
        6
    )

    growth=st.slider(
        "Expected Growth %",
        0,
        30,
        10
    )

    employees=st.number_input(
        "Employees",
        1,
        10000,
        100
    )

    misc=st.number_input(
        "Additional Expenses",
        0,
        100000000,
        500000
    )

    if st.button("Predict Budget"):

        st.info("Prediction Module Coming in Part 2")

# ---------------- PAGE 4 ----------------

elif page=="Dashboard":

    st.header("📈 Dashboard")

    col1,col2,col3=st.columns(3)

    col1.metric(
        "Predicted Budget",
        "₹2.4 Cr",
        "+12%"
    )

    col2.metric(
        "Marketing",
        "₹40 L",
        "+8%"
    )

    col3.metric(
        "Salary",
        "₹1.3 Cr",
        "+15%"
    )

    data=pd.DataFrame({

        "Department":[

            "Salary",

            "Marketing",

            "Rent",

            "IT",

            "Misc"

        ],

        "Budget":[

            13000000,

            4000000,

            1200000,

            900000,

            600000

        ]

    })

    fig=px.bar(

        data,

        x="Department",

        y="Budget",

        color="Department",

        title="Department Budget"

    )

    st.plotly_chart(fig,use_container_width=True)
  streamlit run app.py

import pandas as pd

def load_data(uploaded_files):
    all_data = []

    for file in uploaded_files:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        else:
            df = pd.read_excel(file)

        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    return data


def clean_data(df):

    df = df.drop_duplicates()

    df = df.fillna(0)

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    return df
import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_PATH = "models/budget_model.pkl"

def train_budget_model(df):
    """
    Trains a Linear Regression model using historical budget data.
    Target column must be 'Total Budget'.
    """

    # Ensure model folder exists
    os.makedirs("models", exist_ok=True)

    # Features
    X = df.drop(columns=["Total Budget"])

    # Target
    y = df["Total Budget"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return model


def load_budget_model():
    """Loads trained model if available."""

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return None
  import pandas as pd

def predict_budget(model, user_input):

    prediction = model.predict(user_input)

    return prediction[0]
