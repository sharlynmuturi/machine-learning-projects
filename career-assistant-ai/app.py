import streamlit as st
import requests
import pandas as pd
import pdfplumber
import chromadb
import uuid
import os
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Career Assistant", layout="wide")
st.title("AI Resume & Cover Letter Tailoring")

BASE_DIR = Path(__file__).parent

# Load resume
resume_path = BASE_DIR / "resume.pdf"

def read_resume(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

resume_text = read_resume(resume_path)
st.success("Resume loaded from project directory")


# ChromaDB setup
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

# Load your portfolio 
portfolio_path = BASE_DIR / "portfolio.csv"
portfolio_df = pd.read_csv(portfolio_path)

# Add projects to ChromaDB if empty
if not collection.count():
    for _, row in portfolio_df.iterrows():
        collection.add(
            documents=[row["all_text"]],  # vectorizing the full text
            metadatas={
                "link": row.get("link", ""),
                "project_name": row.get("project_name", ""),
                "tech_stack": row.get("tech_stack", "")  # empty string if missing
            },
            ids=[str(uuid.uuid4())]
        )
        
st.info(f"ChromaDB loaded with {collection.count()} portfolio projects")


# Job Link
job_link = st.text_input("Enter Job Description URL")


# Scrape job posting
def scrape_job_page(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    if docs:
        return docs[0].page_content  # get text of the first page
    return ""
    
job_data = None
if job_link:
    job_data = scrape_job_page(job_link)

# LLM Setup
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

parser = JsonOutputParser()

# Retrieve top portfolio projects
def get_top_projects(job_text, n=7):
    results = collection.query(
        query_texts=[job_text],
        n_results=n
    )
    top_projects_text = ""
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        project_name = meta.get("project_name", "Unknown Project")
        tech_stack = meta.get("tech_stack", "")  # empty string if missing
        link = meta.get("link", "")
        
        top_projects_text += f"""
Project: {project_name}
Description: {doc}
Technologies: {tech_stack}
Link: {link}
"""
    return top_projects_text

top_projects_text = None
if job_data:
    top_projects_text = get_top_projects(job_data)
    st.subheader("Top Portfolio Projects")
    st.text_area("Relevant Projects", top_projects_text, height=200)


# Prompt Templates
prompt_resume = PromptTemplate.from_template(
"""
You are an expert resume parser.

### TASK
Extract structured information from the resume text.

Return a JSON object with the following fields:

- name: candidate name
- education: list of education entries
- skills: list of technical skills
- experience: list of professional experiences
- projects: list of projects mentioned

### RULES
- Extract information only from the text.
- Do NOT invent information.
- If something is missing return null.
- Skills must be a list of strings.
- Return ONLY valid JSON.

### JSON FORMAT
{{
  "name": "Candidate name",
  "education": ["degree1", "degree2"],
  "skills": ["skill1", "skill2"],
  "experience": ["experience1", "experience2"],
  "projects": ["project1", "project2"]
}}

### RESUME TEXT
{resume_text}
"""
)

chain_resume = prompt_resume | llm | parser

# Running the resume extraction
resume_data = chain_resume.invoke({"resume_text": resume_text})


prompt_resume_tailor = PromptTemplate.from_template("""
You are an expert career assistant and resume writer.

Your task is to tailor a candidate's resume for a specific job role.

### JOB DESCRIPTION
{job_data}

### CANDIDATE RESUME
{resume_data}

### CANDIDATE PROJECTS
{portfolio_projects}

### INSTRUCTIONS
Rewrite the candidate's resume so that it better aligns with the job.

Focus on:
- highlighting relevant skills
- emphasizing relevant projects
- matching the language used in the job description
- keeping all information truthful

### OUTPUT FORMAT

Return a professional resume with the following sections:

NAME

PROFESSIONAL SUMMARY

SKILLS

EXPERIENCE

PROJECTS

EDUCATION

The resume should be concise, professional, and optimized for ATS systems.

Return only the resume text.
""")

prompt_cover_letter = PromptTemplate.from_template("""
You are an expert career assistant.

### CONTEXT
You have the following information:

- Job posting details: {job_data}
- Candidate resume content: {resume_data}
- Relevant portfolio projects: {portfolio_projects}

### TASK
Write a professional and persuasive **cover letter** tailored to this specific job. 
The cover letter should:

1. Address the hiring manager (use "Dear Hiring Manager" if name is unknown)
2. Highlight the candidate's most relevant skills and experience from the resume
3. Reference the most relevant portfolio projects
4. Match the tone of the job posting (formal, professional)
5. Be 3–5 paragraphs long
6. End with a polite call-to-action

### RULES
- Only use information provided in the context; do not invent details
- Keep it concise and impactful
- Output plain text (no JSON)
""")


# Buttons
if job_data:
    if st.button("Tailor Resume"):
        with st.spinner("Tailoring resume..."):
            res_resume = chain_resume.invoke({...})
        chain_resume = prompt_resume_tailor | llm
        res_resume = chain_resume.invoke({
            "job_data": job_data,
            "resume_data": resume_text,
            "portfolio_projects": top_projects_text
        })
        tailored_resume = res_resume["content"] if isinstance(res_resume, dict) else res_resume.content
        st.subheader("Tailored Resume")
        st.text_area("Tailored Resume", tailored_resume, height=400)

    if st.button("Generate Cover Letter"):
        with st.spinner("Generating Cover Letter..."):
            res_resume = chain_resume.invoke({...})
        chain_cover = prompt_cover_letter | llm
        res_cover = chain_cover.invoke({
            "job_data": job_data,
            "resume_data": resume_text,
            "portfolio_projects": top_projects_text
        })
        cover_letter = res_cover["content"] if isinstance(res_cover, dict) else res_cover.content
        st.subheader("AI Generated Cover Letter")
        st.text_area("Cover Letter", cover_letter, height=400)
