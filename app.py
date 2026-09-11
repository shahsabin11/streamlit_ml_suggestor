import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("Dataset ML Model Suggester & Statistical Forecaster")
st.write("Upload a CSV file, select your target variable, and let OpenAI estimate the performance metrics for top models.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())
    
    target_column = st.selectbox("Select your response (target) variable", df.columns)
    
    if st.button("Generate Model Forecast & Summary"):
        dtypes_str = df.dtypes.to_string()
        shape = df.shape
        missing_vals = df.isnull().sum().to_string()
        
        prompt = f"""
        You are an expert data scientist and statistical analyst. 
        Dataset summary:
        - Shape: {shape[0]} rows, {shape[1]} columns
        - Target response variable: '{target_column}'
        - Data types:
        {dtypes_str}
        - Missing values summary:
        {missing_vals}
        
        Based on the structure, size, and data types of this dataset, suggest 5 appropriate machine learning models. 
        
        For each model, provide a simulated statistical performance summary as if the model had just been trained on this data. Include:
        1. **Model Name**
        2. **Estimated Accuracy / Fit Score:** A clear predicted accuracy percentage (for classification) or R-squared / goodness-of-fit score (for regression).
        3. **Pros & Cons:** One clear Pro and one clear Con for this dataset.
        4. **Statistical Signifiers:** Estimated p-values, F-statistic, or coefficients significance context where applicable.
        
        Format your response clearly with Markdown headings or bullet points for each model.
        """
        
        with st.spinner("OpenAI is analyzing your dataset and simulating model performance summaries..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            st.subheader("AI-Generated Model Performance Forecasts")
            st.markdown(response.choices[0].message.content)