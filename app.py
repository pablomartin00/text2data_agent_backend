import streamlit as st
import requests
import pandas as pd

# URL of the Flask server
FLASK_SERVER_URL = "http://127.0.0.1:5000"

# Initialize session on the Flask server
def init_session():
    response = requests.post(f"{FLASK_SERVER_URL}/init_session")
    if response.status_code == 200:
        data = response.json()
        st.session_state["session_id"] = data["session_id"]
        load_data()

# Load data from the Flask server
def load_data():
    response = requests.get(f"{FLASK_SERVER_URL}/get_data")
    
    if response.status_code == 200:
        data = response.json()
        # Specify the desired column order
        column_order = ['target_organization_name', 'subject', 'created_by_name', 'content']
        df = pd.DataFrame(data)[column_order]
        st.session_state["data"] = df

# Ask a question to the expert
def ask_expert():
    question = st.session_state.question_input
    if "session_id" not in st.session_state:
        st.write("Session not initialized")
        return
    
    with st.spinner("Elaborating response..."):
        response = requests.post(f"{FLASK_SERVER_URL}/ask_expert", json={
            "session_id": st.session_state["session_id"],
            "question": question
        })
        
        if response.status_code == 200:
            data = response.json()
            st.session_state["conversation"].insert(0, {
                'user': question,
                'expert': data['expert_response']
            })
            st.session_state.question_input = ""  # Clear the input field

# Display the current conversation
def get_conversation():
    if "session_id" not in st.session_state:
        st.write("Session not initialized")
        return
    
    response = requests.get(f"{FLASK_SERVER_URL}/conversation", params={
        "session_id": st.session_state["session_id"]
    })
    
    if response.status_code == 200:
        data = response.json()
        conversation = data['conversation']

# Automatically initialize session when loading the app
if "session_initialized" not in st.session_state:
    init_session()
    st.session_state["session_initialized"] = True

# Ensure necessary keys are initialized in session_state
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Move title a bit higher
st.markdown("<h1 style='margin-top: -50px;'>Notes Expert Chat</h1>", unsafe_allow_html=True)

if "data" in st.session_state:
    
     # Display the dataframe with limited height
     with st.expander("Notes Data", expanded=True):
         st.dataframe(st.session_state["data"], height=150)

     # Chat window for expert Q&A
     with st.expander("Notes Expert", expanded=True):
         question_input = st.text_input("Ask your notes:", key="question_input", on_change=ask_expert)
         
         if st.button("Ask Expert"):
             ask_expert()
         
         if len(st.session_state["conversation"]) > 0:
             for entry in st.session_state["conversation"]:
                 user_question, expert_response = entry.values()
                 st.write(f"**You:** {user_question}")
                 st.write(f"**Expert:** {expert_response}")