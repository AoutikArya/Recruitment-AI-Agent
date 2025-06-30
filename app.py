import streamlit as st
from Recruitment_agentic_workflow import extract_text_from_pdf, parse_resume_with_llm
from Recruitment_agentic_workflow import setup_workflow
import json

def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("📄 Agentic Recruitment Workflow")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Enter the job role:")
    with col2:
        uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")
    
    if st.button("Screen Candidate") and uploaded_file and role:
        with st.spinner("Processing..."):
            try:
                # Parse resume
                text = extract_text_from_pdf(uploaded_file)
                parsed_data = parse_resume_with_llm(text)
                
                # Run screening
                app = setup_workflow()
                results = app.invoke({
                    "application": parsed_data,
                    "role": role
                })
                
                # Display results
                st.subheader("Results")
                with st.expander("Parsed Resume Data"):
                    st.json(parsed_data)
                
                st.metric("Experience Level", results['experience_level'])
                st.metric("Skill Match", results['skill_match'])
                
                st.success(results['response'])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()