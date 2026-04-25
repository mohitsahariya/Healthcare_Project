
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Insurance Cost Analytics Dashboard")

df = pd.read_csv("../data/insurance.csv")

st.sidebar.header("Filters")
smoker = st.sidebar.selectbox("Smoker", df["smoker"].unique())
region = st.sidebar.selectbox("Region", df["region"].unique())

filtered_df = df[(df["smoker"] == smoker) & (df["region"] == region)]

st.metric("Average Charges", round(filtered_df["charges"].mean(), 2))

fig1 = px.scatter(filtered_df, x="bmi", y="charges", color="age")
st.plotly_chart(fig1)

fig2 = px.bar(filtered_df, x="children", y="charges", color="sex")
st.plotly_chart(fig2)
