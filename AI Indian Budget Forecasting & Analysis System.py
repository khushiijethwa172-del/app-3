import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(
    page_title="AI Budget Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Budget Prediction System")
st.write("Upload previous Indian Budget PDFs and predict next year's budget.")
uploaded_files = st.file_uploader(
    "Upload Budget PDFs",
    type="pdf",
    accept_multiple_files=True
)
categories = [
    "Agriculture",
    "Education",
    "Health",
    "Defence",
    "Railways",
    "Infrastructure",
    "MSME",
    "Rural Development",
    "Women",
    "Tax Revenue"
]
@st.cache_data
def extract_pdf(pdf):

    rows = []

    with pdfplumber.open(pdf) as pdf_file:

        for page in pdf_file.pages:

            text = page.extract_text()

            if text:

                for line in text.split("\n"):

                    for category in categories:

                        if category.lower() in line.lower():

                            numbers = re.findall(r'[\d,]+', line)

                            if numbers:

                                amount = numbers[-1].replace(",", "")

                                rows.append([category, float(amount)])

    return rows
    all_data = []

if uploaded_files:

    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):

        year = re.findall(r"\d{4}", file.name)

        if year:
            year = int(year[0])
        else:
            year = 2020 + i

        extracted = extract_pdf(file)

        for row in extracted:

            all_data.append(
                {
                    "Year": year,
                    "Budget Category": row[0],
                    "Budget Allocation": row[1]
                }
            )

        progress.progress((i + 1) / len(uploaded_files))
        if len(all_data) > 0:

    df = pd.DataFrame(all_data)

    st.success("✅ Budget Data Extracted Successfully")

    st.subheader("Extracted Dataset")

    st.dataframe(df)
        df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    df["Budget Allocation"] = pd.to_numeric(
        df["Budget Allocation"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    st.subheader("Clean Dataset")

    st.dataframe(df)
    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))

    col2.metric(
        "Categories",
        df["Budget Category"].nunique()
    )

    col3.metric(
        "Years",
        df["Year"].nunique()
    )
    st.subheader("Budget Trend")

    fig = px.line(
        df,
        x="Year",
        y="Budget Allocation",
        color="Budget Category",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    latest = df.groupby(
        "Budget Category"
    )["Budget Allocation"].mean().reset_index()

    fig = px.bar(
        latest,
        x="Budget Category",
        y="Budget Allocation",
        color="Budget Category"
    )

    st.plotly_chart(fig, use_container_width=True)
    fig = px.pie(
        latest,
        values="Budget Allocation",
        names="Budget Category"
    )

    st.plotly_chart(fig, use_container_width=True)
st.subheader("🤖 AI Budget Prediction")

predictions = []

for category in df["Budget Category"].unique():

    temp = df[df["Budget Category"] == category]

    if len(temp) >= 2:

        X = temp[["Year"]]

        y = temp["Budget Allocation"]

        model = LinearRegression()

        model.fit(X, y)

        next_year = np.array([[temp["Year"].max() + 1]])

        prediction = model.predict(next_year)[0]

        predictions.append(
            {
                "Budget Category": category,
                "Predicted Allocation": round(prediction,2)
            }
        )
        prediction_df = pd.DataFrame(predictions)

st.subheader("📈 Next Year Budget Prediction")

st.dataframe(prediction_df)
growth = []

for category in prediction_df["Budget Category"]:

    current = df[df["Budget Category"] == category]

    latest = current.sort_values("Year").iloc[-1]["Budget Allocation"]

    predicted = prediction_df[
        prediction_df["Budget Category"] == category
    ]["Predicted Allocation"].values[0]

    percent = ((predicted-latest)/latest)*100

    growth.append(round(percent,2))

prediction_df["Growth %"] = growth

st.dataframe(prediction_df)
mae_list = []

rmse_list = []

r2_list = []

for category in df["Budget Category"].unique():

    temp = df[df["Budget Category"] == category]

    if len(temp)>=2:

        X=temp[["Year"]]

        y=temp["Budget Allocation"]

        model=LinearRegression()

        model.fit(X,y)

        pred=model.predict(X)

        mae_list.append(mean_absolute_error(y,pred))

        rmse_list.append(np.sqrt(mean_squared_error(y,pred)))

        r2_list.append(r2_score(y,pred))

st.subheader("📊 Model Performance")

col1,col2,col3=st.columns(3)

col1.metric("MAE",round(np.mean(mae_list),2))

col2.metric("RMSE",round(np.mean(rmse_list),2))

col3.metric("R² Score",round(np.mean(r2_list),2))
fig = px.bar(
    prediction_df,
    x="Budget Category",
    y="Predicted Allocation",
    color="Growth %",
    title="Predicted Budget Allocation"
)

st.plotly_chart(fig,use_container_width=True)
csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(

    "⬇ Download Prediction CSV",

    csv,

    "Budget_Prediction.csv",

    "text/csv"

)
st.subheader("📊 Budget Insights")

increase = prediction_df.sort_values(
    "Growth %",
    ascending=False
).head(5)

decrease = prediction_df.sort_values(
    "Growth %",
    ascending=True
).head(5)

col1,col2=st.columns(2)

with col1:

    st.success("Top Increased")

    st.dataframe(increase)

with col2:

    st.error("Top Decreased")

    st.dataframe(decrease)
    
