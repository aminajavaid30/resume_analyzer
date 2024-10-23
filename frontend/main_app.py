import streamlit as st
from backend.pdf_ingestion import load_split_pdf
from backend.vector_store import create_vector_store
from backend.analysis import analyze_resume
import os
import shutil

# Main application including "Upload Resume" and "Resume Analysis" sections - displayed at the left
def render_main_app():

    col1, col2 = st.columns([1, 2])  # Creating two columns for layout

    with col2:
        st.header("Resume Analysis")  # Header for the analysis section

    with col1:
        st.header("Upload Resume")  # Header for the upload section
        
        # File uploader for PDF resumes
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

        # Text area for job description input
        job_description = st.text_area("Enter Job Description", height=300)

        if resume_file and job_description:  # Check if both inputs are provided
            # Create a temporary directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary directory
            with open(os.path.join(temp_dir, resume_file.name), "wb") as f:
                f.write(resume_file.getbuffer())
        
            # Load and split the PDF file into documents and chunks
            resume_file_path = os.path.join("temp", resume_file.name)
            resume_docs, resume_chunks = load_split_pdf(resume_file_path)

            # Create a vector store from the resume chunks
            vector_store = create_vector_store(resume_chunks)
            st.session_state.vector_store = vector_store  # Store vector store in session state
            
            # Remove the temporary directory and its contents
            shutil.rmtree(temp_dir)

            # Button to begin resume analysis
            if st.button("Analyze Resume", help="Click to analyze the resume"):
                # Combine all document contents into one text string for analysis
                full_resume = " ".join([doc.page_content for doc in resume_docs])
                # Anlayze the resume
                analysis = analyze_resume(full_resume, job_description)
                # Store analysis in session state
                st.session_state.analysis = analysis    

            # Display the analysis result if it exists in session state 
            if "analysis" in st.session_state:
                with col2:
                    st.write(st.session_state.analysis)
        else:
            st.info("Please upload a resume and enter a job description to begin.")