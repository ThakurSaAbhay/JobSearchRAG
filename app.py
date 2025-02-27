import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load job metadata
with open("job_metadata.pkl", "rb") as f:
    job_metadata_map = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("🔍 AI-Powered Job Search & Insights : JōbGenie")
st.sidebar.header("🔎 Search Jobs")

# User Input: OpenAI API Key
openai_api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")

# User Input: Job Query
query = st.sidebar.text_area(
    "🎯 Enter job-related question or role (e.g., 'What are top skills for Gen AI jobs?')", 
    "Data Scientist", 
    height=100
)

# Function to search jobs using FAISS
def search_jobs(query, top_k=20):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    jobs = []
    for i in indices[0]:  # Iterate over top job indices
        if i in job_metadata_map:
            jobs.append(job_metadata_map[i])

    return jobs

# Function to generate AI-based insights (uses only 'combined_text' from df)
def generate_gpt_insights(job_df, query, api_key):
    # Extract 'combined_text' for each job
    job_texts = "\n\n".join(job_df["Combined_text"].tolist())

    prompt = (
        f"User searched for: '{query}'\n\n"
        f"You are an AI job assistant. You will be asked a question related to jobs, provide your response based over the context provided only. Do not mention anything about context. "
        f"Here is the context for the give job query:\n{job_texts}"
    )
    openai.api_key=api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a job market analyst."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your OpenAI API key."

if st.sidebar.button("Search"):
    st.sidebar.write("🔍 Searching in FAISS index...")

    job_results = search_jobs(query)

    if not job_results:
        st.error("⚠️ No jobs found. Try a different keyword.")
    else:
        df = pd.DataFrame(job_results)

        # Convert Apply Link to Markdown for Clickable Links
        df["Apply Link"] = df["Job URL"].apply(lambda x: f'<a href="{x}" target="_blank">Apply Here</a>')

        # Display jobs
        st.subheader(f"📌 Top {len(df)} Jobs for '{query}'")
        st.markdown(df[["Title", "Location", "Salary", "Skills", "Apply Link"]].to_markdown(index=False), unsafe_allow_html=True)

        # 📊 **Interactive Job Market Insights**
        st.subheader("📊 Job Market Insights")

        # 🔹 **Interactive Skill Distribution - Pie Chart**
        all_skills = [skill for skills in df["Skills"] for skill in skills.split(", ")]
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        fig_skills = px.pie(
            names=skill_counts.index, 
            values=skill_counts.values, 
            title="Top 10 In-Demand Skills",
            hole=0.3
        )
        st.plotly_chart(fig_skills)

        fig_salary = px.histogram(df, x="Salary", nbins=20, title="Salary Distribution", labels={"Salary": "Salary ($)"})
        st.plotly_chart(fig_salary)

        # 🔹 **Job Locations Heatmap - Treemap**
        location_counts = df["Location"].value_counts().reset_index()
        location_counts.columns = ["Location", "Count"]
        fig_location = px.treemap(
            location_counts, 
            path=["Location"], 
            values="Count", 
            title="Job Locations Breakdown"
        )
        st.plotly_chart(fig_location)


        if "Category" in df.columns:
            category_counts = df["Category"].value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            fig_category = px.sunburst(
                category_counts, 
                path=["Category"], 
                values="Count", 
                title="Job Categories Breakdown"
            )
            st.plotly_chart(fig_category)

        # 📢 **GPT Insights (if API key is provided)**
        if openai_api_key:
            st.subheader("📢 AI-Powered Job Market Analysis")
            insights = generate_gpt_insights(df, query, openai_api_key)
            st.write(insights)
        else:
            st.warning("⚠️ Enter an OpenAI API Key to generate AI insights.")
