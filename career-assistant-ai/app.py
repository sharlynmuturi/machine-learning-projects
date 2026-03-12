import streamlit as st
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
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

resume_text = read_resume(resume_path)
st.success("Resume loaded")


# ChromaDB setup
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

# Loading portfolio 
portfolio_path = BASE_DIR / "portfolio.csv"
portfolio_df = pd.read_csv(portfolio_path)

if not collection.count():
    for _, row in portfolio_df.iterrows():
        collection.add(
            documents=[row["all_text"]],
            metadatas={
                "link": row.get("link",""),
                "project_name": row.get("project_name",""),
                "tech_stack": row.get("tech_stack","")
            },
            ids=[str(uuid.uuid4())]
        )

st.info(f"ChromaDB loaded with {collection.count()} projects")


# LLM setup
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0.3,
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

parser = JsonOutputParser()


# Job link
job_link = st.text_input("Paste Job Description URL")

# Scrape job page
def scrape_job_page(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content if docs else ""

page_data = None
if job_link:
    page_data = scrape_job_page(job_link)


# Job extraction prompt
    
prompt_extract = PromptTemplate.from_template("""
You are an expert information extraction system.

### CONTEXT
The following text was scraped from a company's career page and contains
information about a job posting. The text may include formatting noise,
navigation elements, or repeated content.

### TASK
Extract the job information from the text.

Return a JSON object with the following fields:

- role: The job title
- experience: Required experience level (years or level such as junior, mid, senior)
- skills: List of key technical or domain skills required for the role
- description: A concise 2–5 sentence summary of the job responsibilities

### RULES
- Extract information only from the provided text.
- Do NOT invent information.
- If a field is missing, return null.
- Skills must be returned as a list of strings.
- Return ONLY a valid JSON object.
- Do not include explanations, markdown, backticks, or extra text

### JSON FORMAT
{{
  "role": "Job title",
  "experience": "Experience requirement",
  "skills": ["skill1", "skill2", "skill3"],
  "description": "Short description of the role"
}}

### SCRAPED TEXT
{page_data}
""")

chain_extract = prompt_extract | llm | parser

job_data = None
if page_data:

    with st.spinner("Extracting job details..."):
        job_data = chain_extract.invoke({"page_data": page_data})


# Resume extraction
prompt_resume = PromptTemplate.from_template("""
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
""")

chain_resume_extract = prompt_resume | llm | parser

resume_data = None
with st.spinner("Parsing resume..."):

    @st.cache_data
    def parse_resume(text):
        return chain_resume_extract.invoke({"resume_text": text})

    resume_data = parse_resume(resume_text)

    # resume_data = chain_resume_extract.invoke({"resume_text": resume_text})

# Retrieve portfolio projects
def get_top_projects(job_text, n=7):

    results = collection.query(
        query_texts=[job_text],
        n_results=n
    )

    text = ""

    for meta in results['metadatas'][0]:

        text += f"""
Project: {meta.get("project_name")}
Technologies: {meta.get("tech_stack")}
Link: {meta.get("link")}
"""

    return text

top_projects_text = None

if job_data:
    job_query = job_data["description"] + " " + " ".join(job_data["skills"])

    top_projects_text = get_top_projects(job_query)

    with st.expander("Relevant Portfolio Projects"):
        st.text(top_projects_text)


# Resume tailoring prompt
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


# Cover letter prompt
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
col1, col2 = st.columns(2)

if job_data:

    with col1:

        if st.button("Generate Tailored Resume"):

            with st.spinner("Generating resume..."):

                chain_resume_tailor = prompt_resume_tailor | llm

                res = chain_resume_tailor.invoke({
                    "job_data": job_data,
                    "resume_data": resume_data,
                    "portfolio_projects": top_projects_text
                })

                tailored_resume = res.content

            st.subheader("Tailored Resume")
            st.text_area("", tailored_resume, height=400)

    with col2:

        if st.button("Generate Cover Letter"):

            with st.spinner("Writing cover letter..."):

                chain_cover = prompt_cover_letter | llm

                res = chain_cover.invoke({
                    "job_data": job_data,
                    "resume_data": resume_data,
                    "portfolio_projects": top_projects_text
                })

                cover_letter = res.content

            st.subheader("Cover Letter")
            st.text_area("", cover_letter, height=400)

if page_data:
    st.subheader("Extracted Job Data")
    st.json(job_data)           
