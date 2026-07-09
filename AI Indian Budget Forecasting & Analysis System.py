# ============================================================
# 🇮🇳 AI Budget Prediction System
# Cell 1 - Imports, UI, Dataset Upload & Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)

import joblib
from io import BytesIO
import base64

# --------------------------
# Streamlit Config
# --------------------------

st.set_page_config(
    page_title="🇮🇳 AI Budget Prediction System",
    page_icon="📊",
    layout="wide"
)

# --------------------------
# Custom CSS
# --------------------------

st.markdown("""
<style>

.main{
background:#F5F7FA;
}

.metric-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 2px 10px rgba(0,0,0,.15);
text-align:center;
transition:0.3s;
}

.metric-card:hover{
transform:scale(1.02);
}

.sidebar .sidebar-content{
background:#0B3D91;
}

h1,h2,h3{
color:#0B3D91;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Title
# --------------------------

st.title("🇮🇳 AI Budget Prediction System")

st.caption(
    "Machine Learning based Budget Allocation Prediction Dashboard"
)

# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset Explorer",
        "Visualizations",
        "Machine Learning",
        "Prediction",
        "Analytics"
    ]
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Budget CSV",
    type=["csv"]
)

# --------------------------
# Cache
# --------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --------------------------
# If no file
# --------------------------

if uploaded_file is None:

    st.info("⬅ Upload your Budget CSV from the sidebar.")

    st.stop()

# --------------------------
# Read CSV
# --------------------------

df = load_data(uploaded_file)

# --------------------------
# Auto Detect Columns
# --------------------------

columns = {c.lower(): c for c in df.columns}

year_col = None
sector_col = None
budget_col = None

for c in df.columns:

    x = c.lower()

    if "year" in x:
        year_col = c

    elif "sector" in x or "ministry" in x:
        sector_col = c

    elif "budget" in x or "allocation" in x or "amount" in x:
        budget_col = c

if year_col is None or sector_col is None or budget_col is None:

    st.error("Unable to detect required columns.")

    st.write("CSV must contain:")

    st.write("- Year")

    st.write("- Sector")

    st.write("- Budget_Allocation")

    st.stop()

# --------------------------
# Clean Data
# --------------------------

df = df[[year_col, sector_col, budget_col]]

df.columns = [
    "Year",
    "Sector",
    "Budget"
]

df["Year"] = df["Year"].astype(int)

df["Budget"] = pd.to_numeric(
    df["Budget"],
    errors="coerce"
)

df = df.dropna()

# --------------------------
# Home Page
# --------------------------

if page == "Home":

    latest_year = df["Year"].max()

    latest = df[df["Year"] == latest_year]

    total_budget = latest["Budget"].sum()

    avg_budget = latest["Budget"].mean()

    highest = latest["Budget"].max()

    lowest = latest["Budget"].min()

    sectors = latest["Sector"].nunique()

    years = df["Year"].nunique()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Latest Year",
        latest_year
    )

    c2.metric(
        "Total Budget",
        f"₹ {total_budget:,.0f}"
    )

    c3.metric(
        "Sectors",
        sectors
    )

    c4.metric(
        "Years",
        years
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Average",
        f"{avg_budget:,.0f}"
    )

    c2.metric(
        "Highest",
        f"{highest:,.0f}"
    )

    c3.metric(
        "Lowest",
        f"{lowest:,.0f}"
    )

    st.markdown("---")

    trend = (
        df.groupby("Year")["Budget"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Year",
        y="Budget",
        markers=True,
        title="Budget Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.pie(
        latest,
        names="Sector",
        values="Budget",
        title=f"Budget Distribution ({latest_year})"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# --------------------------
# Dataset Explorer
# --------------------------

elif page == "Dataset Explorer":

    st.subheader("Dataset Preview")

    st.dataframe(df)

    st.write("Rows :", len(df))

    st.write("Columns :", len(df.columns))

    st.write("Missing Values")

    st.write(df.isna().sum())

    st.write("Duplicates")

    st.write(df.duplicated().sum())

    st.write("Data Types")

    st.write(df.dtypes)

    st.write("Summary")

    st.dataframe(df.describe())

    search = st.text_input(
        "Search Sector"
    )

    if search:

        temp = df[
            df["Sector"].str.contains(
                search,
                case=False
            )
        ]

        st.dataframe(temp)

    year = st.selectbox(
        "Filter Year",
        sorted(df["Year"].unique())
    )

    st.dataframe(
        df[df["Year"] == year]
    )
    # ============================================================
# CELL 2
# Visualizations + Machine Learning
# ============================================================

elif page == "Visualizations":

    st.header("📊 Budget Visualizations")

    # ------------------------
    # Year Wise Budget Trend
    # ------------------------

    yearly = (
        df.groupby("Year")["Budget"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        yearly,
        x="Year",
        y="Budget",
        markers=True,
        title="Year Wise Budget Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Top 10 Sectors
    # ------------------------

    latest_year = df["Year"].max()

    latest = df[df["Year"] == latest_year]

    top10 = latest.sort_values(
        "Budget",
        ascending=False
    ).head(10)

    fig = px.bar(
        top10,
        x="Sector",
        y="Budget",
        color="Budget",
        title="Top 10 Sectors"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Bottom 10
    # ------------------------

    bottom10 = latest.sort_values(
        "Budget"
    ).head(10)

    fig = px.bar(
        bottom10,
        x="Sector",
        y="Budget",
        color="Budget",
        title="Bottom 10 Sectors"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Pie Chart
    # ------------------------

    fig = px.pie(
        latest,
        names="Sector",
        values="Budget",
        hole=.45,
        title="Budget Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Histogram
    # ------------------------

    fig = px.histogram(
        df,
        x="Budget",
        nbins=25,
        title="Budget Distribution Histogram"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Box Plot
    # ------------------------

    fig = px.box(
        df,
        y="Budget",
        title="Budget Box Plot"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Scatter Plot
    # ------------------------

    fig = px.scatter(
        df,
        x="Year",
        y="Budget",
        color="Sector",
        title="Scatter Plot"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Area Chart
    # ------------------------

    fig = px.area(
        yearly,
        x="Year",
        y="Budget",
        title="Area Chart"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning")

    encoder = LabelEncoder()

    df["Sector_Code"] = encoder.fit_transform(
        df["Sector"]
    )

    X = df[
        [
            "Year",
            "Sector_Code"
        ]
    ]

    y = df["Budget"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(),

        "Extra Trees":
            ExtraTreesRegressor(),

        "AdaBoost":
            AdaBoostRegressor()

    }

    results = []

    best_model = None

    best_score = -999

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred
        )

        results.append(
            [
                name,
                round(mae,2),
                round(rmse,2),
                round(r2,4)
            ]
        )

        if r2 > best_score:

            best_score = r2

            best_model = model

            best_name = name

        progress.progress(
            (i+1)/len(models)
        )

    results = pd.DataFrame(

        results,

        columns=[
            "Model",
            "MAE",
            "RMSE",
            "R2 Score"
        ]

    )

    st.subheader(
        "Model Performance"
    )

    st.dataframe(results)

    st.success(
        f"🏆 Best Model : {best_name}"
    )

    joblib.dump(
        best_model,
        "saved_model.pkl"
    )

    st.success(
        "Model Saved Successfully"
    )

    fig = px.bar(

        results,

        x="Model",

        y="R2 Score",

        color="R2 Score",

        title="Model Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # ============================================================
# CELL 3
# Prediction + Analytics + Export
# ============================================================

elif page == "Prediction":

    st.header("🔮 Future Budget Prediction")

    try:
        model = joblib.load("saved_model.pkl")
    except:
        st.error("Please train the model first.")
        st.stop()

    future_year = st.number_input(
        "Select Future Year",
        min_value=int(df["Year"].max() + 1),
        value=int(df["Year"].max() + 1),
        step=1
    )

    encoder = LabelEncoder()
    encoder.fit(df["Sector"])

    predictions = []

    latest = df[df["Year"] == df["Year"].max()]

    for sector in encoder.classes_:

        sector_code = encoder.transform([sector])[0]

        pred = model.predict(
            [[future_year, sector_code]]
        )[0]

        previous = latest[
            latest["Sector"] == sector
        ]["Budget"]

        previous_budget = (
            previous.iloc[0]
            if len(previous) > 0
            else 0
        )

        growth = (
            (pred - previous_budget)
            / previous_budget * 100
            if previous_budget != 0
            else 0
        )

        predictions.append([
            sector,
            previous_budget,
            pred,
            growth
        ])

    prediction_df = pd.DataFrame(
        predictions,
        columns=[
            "Sector",
            "Previous Budget",
            "Predicted Budget",
            "Growth %"
        ]
    )

    prediction_df["Trend"] = np.where(
        prediction_df["Growth %"] >= 0,
        "📈 Increase",
        "📉 Decrease"
    )

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    csv = prediction_df.to_csv(index=False)

    st.download_button(
        "⬇ Download CSV",
        csv,
        "Budget_Prediction.csv",
        "text/csv"
    )

    excel = BytesIO()

    with pd.ExcelWriter(
        excel,
        engine="openpyxl"
    ) as writer:

        prediction_df.to_excel(
            writer,
            index=False
        )

    st.download_button(
        "⬇ Download Excel",
        excel.getvalue(),
        "Budget_Prediction.xlsx"
    )

# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.header("📈 Budget Analytics")

    latest = df[df["Year"] == df["Year"].max()]

    c1, c2 = st.columns(2)

    highest = latest.loc[
        latest["Budget"].idxmax()
    ]

    lowest = latest.loc[
        latest["Budget"].idxmin()
    ]

    c1.metric(
        "Highest Funded Sector",
        highest["Sector"]
    )

    c2.metric(
        "Lowest Funded Sector",
        lowest["Sector"]
    )

    st.metric(
        "Average Budget",
        f"₹ {latest['Budget'].mean():,.0f}"
    )

    st.metric(
        "Median Budget",
        f"₹ {latest['Budget'].median():,.0f}"
    )

    st.metric(
        "Standard Deviation",
        f"{latest['Budget'].std():,.2f}"
    )

    growth = (
        df.groupby("Sector")
        .apply(
            lambda x:
            (
                x.sort_values("Year")
                .iloc[-1]["Budget"]
                -
                x.sort_values("Year")
                .iloc[0]["Budget"]
            )
        )
        .reset_index(name="Growth")
    )

    fig = px.bar(
        growth.sort_values(
            "Growth",
            ascending=False
        ),
        x="Sector",
        y="Growth",
        color="Growth",
        title="Sector Growth"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("🏆 Budget Ranking")

    rank = latest.sort_values(
        "Budget",
        ascending=False
    )

    rank["Rank"] = range(
        1,
        len(rank) + 1
    )

    st.dataframe(rank)

    st.subheader("💡 AI Insights")

    st.success(
        f"""
• Highest funded sector is **{highest['Sector']}**

• Lowest funded sector is **{lowest['Sector']}**

• Total Budget :
₹ {latest['Budget'].sum():,.0f}

• Average Allocation :
₹ {latest['Budget'].mean():,.0f}

• The trained model can now be used to forecast future Union Budgets.
"""
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
"""
🇮🇳 AI Budget Prediction System

Built using Streamlit • Plotly • Scikit-Learn • Pandas
"""
)
encoder = LabelEncoder()
encoder.fit(df["Sector"])
df["Sector_Code"] = encoder.transform(df["Sector"])

joblib.dump(encoder, "label_encoder.pkl")
encoder = LabelEncoder()
encoder.fit(df["Sector"])
encoder = joblib.load("label_encoder.pkl")
