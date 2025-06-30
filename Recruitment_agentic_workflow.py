
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from io import BytesIO

llm = ChatGroq(model='llama-3.3-70b-versatile') 

# Step 1: Define your output schema


class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    contact: dict = Field(description="Contact information including email, phone, LinkedIn")
    experience: list = Field(description="List of work experiences with job titles, companies, and durations")
    skills: list[str] = Field(description="Technical and professional skills")
    education: list = Field(description="Educational background with degrees and institutions")

# Step 2: PDF Text Extractor
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    # Create a file-like object from the uploaded file
    file_bytes = BytesIO(uploaded_file.read())
    reader = PdfReader(file_bytes)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 3: LLM Processing Pipeline
def parse_resume_with_llm(resume_text: str) -> dict:
    # Set up the LLM
    llm = ChatGroq(model='llama-3.3-70b-versatile')  # Use gpt-4-turbo for best results
    
    # Create the processing chain
    parser = JsonOutputParser(pydantic_object=Resume)
    
    prompt = ChatPromptTemplate.from_template(
        """Extract the following information from the resume below. 
        Return ONLY valid JSON matching the specified schema.
        
        Resume:
        {resume_text}
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    
    return chain.invoke({
        "resume_text": resume_text,
        "format_instructions": parser.get_format_instructions()
    })

# Step 4: Main Execution
def process_resume_pdf(pdf_path: str) -> dict:
    try:
        # Extract raw text
        resume_text = extract_text_from_pdf(pdf_path)
        
        # Process with LLM
        parsed_data = parse_resume_with_llm(resume_text)
        
        return {
            "success": True,
            "data": parsed_data,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

# pdf_path=r"C:\Users\Aoutik Arya\Downloads\Aoutik_Arya_cv- (1).pdf"
# result = process_resume_pdf(pdf_path)



from typing_extensions import TypedDict
class State(TypedDict):
  application: dict
  role: str 
  experience_level: str
  skill_match : str
  response: str




def categorize_experience(state: State) -> State:
  print("\nCategorizing the experience level of candidate : ")
  prompt = ChatPromptTemplate.from_template(
    f"Today's date = {datetime.today()} , use it calculate experience in years."
      "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'"
      "Respond with either  'Entry-level', 'Mid-level' or 'Senior-level' only no extra text."
      "Application : {application}"
  )
  chain = prompt | llm
  experience_level = chain.invoke({"application": state["application"]}).content
  print(f"Experience Level : {experience_level}")
  return {"experience_level" : experience_level}

def assess_skillset(state: State) -> State:
  print("\nAssessing the skillset of candidate : ")
  prompt = ChatPromptTemplate.from_template(
      f"Based on the job application for a {state['role']}, assess the candidate's skillset"
        "strictly check for required skills only."
       "Respond with either 'Match' or 'No Match' only, no extra text."
      "Application : {application}"
  )
  chain = prompt | llm
  skill_match = chain.invoke({"application": state["application"]['skills']}).content
  print(f"Skill Match : {skill_match}")
  return {"skill_match" : skill_match}

def schedule_hr_interview(state: State) -> State:
  print("\nScheduling the interview : ")
  return {"response" : "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
  print("Escalating to recruiter")
  return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
  print("Sending rejecting email")
  return {"response" : f"Candidate doesn't meet JD and has been rejected, Sending rejection mail at {state['application']['contact']['email']} "}


def route_app(state: State) :
  if state['skill_match']=='Match':
    return "schedule_hr_interview"
  elif state['skill_match']!="Match" and state["experience_level"] == "Senior-level":
    
    return "escalate_to_recruiter"
  else:
    return "reject_application"


from langgraph.graph import StateGraph, START, END
def setup_workflow():
    workflow = StateGraph(State)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges("assess_skillset", route_app,
                                {'schedule_hr_interview':"schedule_hr_interview",
                                    "reject_application":'reject_application',
                                    "escalate_to_recruiter":"escalate_to_recruiter"})
    workflow.add_edge(START, "categorize_experience")

    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)
    workflow.add_edge("schedule_hr_interview", END)


    return workflow.compile()

def run_candidate_screening(application: dict):
  results = app.invoke({"application" : application})
  return {
      "experience_level" : results["experience_level"],
      "skill_match" : results["skill_match"],
      "response" : results["response"]
  }


# application_text = result['data']
# results = run_candidate_screening(application_text)
# print("\n\nComputed Results :")
# print(f"Application: {application_text}")
# print(f"Experience Level: {results['experience_level']}")
# print(f"Skill Match: {results['skill_match']}")
# print(f"Response: {results['response']}")





def main():
    st.title("AI Resume Screener")
    st.write("Upload a resume PDF and enter the job role to screen candidates")
    
    # Get job role
    role = st.text_input("Enter required role:")
    st.session_state.role = role
    
    # File upload
    uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")
    
    if uploaded_file and role:
        if st.button("Process Resume"):
            with st.spinner("Processing resume..."):
                try:
                    # Extract and parse resume
                    text = extract_text_from_pdf(uploaded_file)
                    parsed_data = parse_resume_with_llm(text)
                    
                    # Run screening workflow
                    
                    results = app.invoke({"application": parsed_data['data']})
                    
                    # Display results
                    st.subheader("Screening Results")
                    st.json(parsed_data)
                    
                    st.write(f"**Experience Level:** {results['experience_level']}")
                    st.write(f"**Skill Match:** {results['skill_match']}")
                    st.write(f"**Decision:** {results['response']}")
                    
                except Exception as e:
                    st.error(f"Error processing resume: {str(e)}")

if __name__ == "__main__":
    main()