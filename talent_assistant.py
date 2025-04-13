import streamlit as st
from crewai import Agent, Task
from langchain.tools import tool
import google.generativeai as genai  # Changed from ChatOpenAI
import chromadb
from chromadb.utils import embedding_functions
import json
import PyPDF2
import docx
import os
from dotenv import load_dotenv
import random
import re

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Changed from OPENAI_API_KEY


# ========================
# Database Configuration
# ========================
class ChromaDBManager:
    def __init__(self):
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="candidates",
            embedding_function=self.embedding_function
        )

    def add_candidate(self, candidate_data: dict):
        try:
            document = json.dumps(candidate_data) if not isinstance(candidate_data, str) else candidate_data
            self.collection.add(
                documents=[document],
                ids=[str(len(self.collection.get()["ids"]) + 1)]
            )
            return True
        except Exception as e:
            st.error(f"Error adding candidate: {str(e)}")
            return False

    def get_all_candidates(self):
        try:
            all_data = self.collection.get()
            return [json.loads(doc) for doc in all_data["documents"]]
        except Exception as e:
            st.error(f"Error retrieving candidates: {str(e)}")
            return []


# ========================
# LLM Configuration (Gemini)
# ========================
def setup_llm():
    try:
        # Attempt to use a potentially available Gemini Pro model
        model = genai.GenerativeModel('gemini-pro')
        # Test if generate_content is supported (a basic check)
        _ = model.generate_content("Test")
        return model
    except Exception as e:
        print(f"Error initializing gemini-pro: {e}")
        print("Trying 'gemini-pro-vision' as an alternative...")
        try:
            model = genai.GenerativeModel('gemini-pro-vision')
            _ = model.generate_content("Test")
            return model
        except Exception as e:
            print(f"Error initializing gemini-pro-vision: {e}")
            print("Trying 'gemini-1.5-pro' as an alternative...")
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                # Need to check supported methods properly, but this is a basic attempt
                return model
            except Exception as e:
                raise Exception(f"Failed to initialize Gemini models: {e}")


# ========================
# Agent Implementations
# ========================
class SourcingAgent:
    def __init__(self, llm, db_manager):
        self.llm = llm
        self.db_manager = db_manager
        self.agent = Agent(
            role='Sourcing Specialist',
            goal='Find qualified candidates',
            backstory='Expert in candidate sourcing from various platforms',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def search_job_boards(self, skill: str, min_experience: int) -> str:
        first_names = ["Arjun", "Priya", "Rohan", "Neha", "Vikram", "Anita", "Rahul", "Sneha", "Karan", "Meera",
                       "Deepak", "Isha", "Amit", "Pooja", "Raj", "Kavita", "Nikhil", "Divya", "Sanjay", "Tina",
                       "Aditya", "Anjali", "Ravi", "Simran", "Akash", "Komal", "Manoj", "Ritu", "Sunil", "Preeti",
                       "Varun", "Shalini", "Yash", "Sonia", "Abhishek", "Sakshi", "Ajay", "Nisha", "Harsh", "Payal",
                       "Tarun", "Aarti", "Vivek", "Bhavna", "Suresh", "Kiran", "Mohit", "Alka", "Hemant", "Geeta"]

        domains = ["example.com", "mail.com", "demo.org", "techjobs.io", "workhub.net"]

        mock_data = [
            {
                "name": f"{first_names[i]}",
                "email": f"{first_names[i].lower()}{i}@{random.choice(domains)}",
                "skills": ["Java", "Spring"] if i % 2 == 0 else ["Python", "SQL"],
                "experience": (i % 11) + 1,
                "status": "New"
            } for i in range(50)
        ]

        filtered = [c for c in mock_data if
                    skill.lower() in [s.lower() for s in c["skills"]] and c["experience"] >= min_experience]
        return json.dumps(filtered)

    def run(self):
        st.header("🔍 Candidate Sourcing")
        with st.form("sourcing_form"):
            skill = st.text_input("Skill (e.g., Java, Python)", "Java")
            min_experience = st.slider("Minimum Experience (Years)", 0, 15, 5)
            if st.form_submit_button("Search"):
                try:
                    results = json.loads(self.search_job_boards(skill, min_experience))
                    st.session_state.candidates = results
                    st.session_state.page = 0
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

        if 'candidates' in st.session_state:
            candidates = st.session_state.candidates
            page = st.session_state.get("page", 0)
            per_page = 5
            total_pages = (len(candidates) - 1) // per_page + 1

            st.subheader(f"Search Results (Page {page + 1} of {total_pages})")

            start_idx = page * per_page
            end_idx = min(start_idx + per_page, len(candidates))

            for candidate in candidates[start_idx:end_idx]:
                with st.expander(f"{candidate['name']} - {', '.join(candidate['skills'])}"):
                    st.write(f"Experience: {candidate['experience']} years")
                    st.write(f"Email: {candidate['email']}")
                    if st.button("Add to Pipeline", key=f"{candidate['name']}"):
                        if self.db_manager.add_candidate(candidate):
                            st.success(f"Added {candidate['name']}!")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⬅️ Previous") and page > 0:
                    st.session_state.page -= 1
            with col2:
                if st.button("Next ➔") and page < total_pages - 1:
                    st.session_state.page += 1


class ScreeningAgent:
    def __init__(self, llm, db_manager):
        self.llm = llm
        self.db_manager = db_manager

    def parse_resume(self, file_path: str) -> str:
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join([page.extract_text() for page in reader.pages])
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            return "Unsupported file format"
        except Exception as e:
            return f"Error parsing file: {str(e)}"

    def basic_resume_analysis(self, text: str) -> str:
        lines = text.splitlines()
        email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
        found_email = email_match.group() if email_match else "Not found"
        name = lines[0] if lines else "Unknown"

        keywords = ["Python", "Java", "SQL", "Machine Learning", "Django", "Spring"]
        found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]

        analysis = f"**Name:** {name}\n"
        analysis += f"**Email:** {found_email}\n"
        analysis += f"**Detected Skills:** {', '.join(found_keywords) if found_keywords else 'None found'}\n"
        analysis += f"**Match Score:** {min(len(found_keywords) * 20, 100)}%"

        return analysis, name, found_email, found_keywords

    def run(self):
        st.header("📄 Resume Screening")
        uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

        if uploaded_file:
            file_path = f"temp_{uploaded_file.name}"
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                text = self.parse_resume(file_path)
                st.subheader("Extracted Text")
                st.text_area("Resume Content", text, height=300)

                if st.button("Analyze Resume"):
                    with st.spinner("Analyzing..."):
                        result, name, found_email, found_keywords = self.basic_resume_analysis(text)
                        st.markdown(result)

                        candidate_data = {
                            "name": name,
                            "email": found_email,
                            "skills": found_keywords,
                            "experience": 0,
                            "status": "Screened"
                        }
                        self.db_manager.add_candidate(candidate_data)

                        st.success("Analysis complete and saved!")
            except Exception as e:
                st.error(f"File processing error: {str(e)}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)


class PipelineViewer:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def run(self):
        st.header("📋 Candidate Pipeline")
        candidates = self.db_manager.get_all_candidates()

        if not candidates:
            st.info("No candidates in the pipeline yet.")
            return

        for candidate in candidates:
            with st.expander(f"{candidate['name']} - {candidate['email']}"):
                st.write(f"Skills: {', '.join(candidate['skills'])}")
                st.write(f"Experience: {candidate['experience']} years")
                st.write(f"Status: {candidate['status']}")


# ========================
# Chatbot Module (Gemini-powered)
# ========================
def run_chatbot():
    st.header("🤖 Talent Assistant Chatbot ")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    example_prompts = [
        "List all the candidates currently in the pipeline.",
        "Tell me the names and emails of candidates with Python skills.",
        "Are there any candidates with more than 5 years of experience?",
        "Summarize the skills of the candidate named Priya.",
        "How many candidates have 'New' status?",
        "Which candidates have both Java and Spring skills?",
        "What is the average experience level of the candidates?",
        "Identify candidates who might be a good fit for a senior role based on experience.",
        "Give me the contact information of all screened candidates.",
        "Are there any candidates with Machine Learning skills?",
        "People who know Java with 1 year experience.",
        "People who know Java with 5 years experience.",
        "People who know Java with 8 years experience.",
        "People who know Python with 1 year experience.",
        "People who know Python with 5 years experience."
    ]

    st.subheader("Try these prompts:")
    cols = st.columns(5)
    for i, prompt in enumerate(example_prompts):
        with cols[i % 5]:
            if st.button(prompt):
                st.session_state.user_input = prompt

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("You", user_input))

        try:
            model = setup_llm()
            db_manager = ChromaDBManager()
            candidates = db_manager.get_all_candidates()
            prompt = f"Based on the following candidates: {json.dumps(candidates)}, answer: {user_input} " \
                     f"When you mention candidate(s), please provide their name, email, skills, and experience."

            response = model.generate_content(prompt)
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"Model error in chatbot: {str(e)}"

        st.session_state.chat_history.append(("Bot", bot_reply))

    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")


# ========================
# Main Application
# ========================
def main():
    st.set_page_config(page_title="Talent Acquisition Assistant", layout="wide")
    st.title("Intelligent Talent Acquisition Assistant ")

    try:
        llm = setup_llm()
        db_manager = ChromaDBManager()

        menu = st.sidebar.selectbox(
            "Select Module",
            ["Sourcing", "Screening", "Pipeline", "Chatbot"]
        )

        if menu == "Sourcing":
            SourcingAgent(llm, db_manager).run()
        elif menu == "Screening":
            ScreeningAgent(llm, db_manager).run()
        elif menu == "Pipeline":
            PipelineViewer(db_manager).run()
        elif menu == "Chatbot":
            run_chatbot()

    except Exception as e:
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Missing GEMINI_API_KEY in .env file")
    else:
        main()