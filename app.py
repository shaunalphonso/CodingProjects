import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize environment variables
load_dotenv()

# Check and store OpenAI API key
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY is not set")
    exit(1)
else:
    print("OPENAI_API_KEY is set")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None

if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Store API token in session state
st.session_state.openai_key = os.getenv("OPENAI_API_KEY")

# Function to find the most relevant rows based on the question
def find_relevant_rows(question, df, num_rows=1):
    documents = [question] + list(df.to_string(header=False, index=False))
    vectorizer = TfidfVectorizer().fit_transform(documents)
    cosine_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    most_similar_row_index = cosine_matrix.argsort()[0][-num_rows:][::-1]
    
    # Check if we have any similar rows, else use the entire DataFrame
    if most_similar_row_index.size > 0:
        most_similar_rows = df.iloc[most_similar_row_index]
    else:
        most_similar_rows = df
    
    return most_similar_rows

# Function to ask the OpenAI chat model
def ask_openai_chat_model(question, context):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4-32k",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {st.session_state.openai_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

# Streamlit Interface
st.title("Ask your CSV 📈")

# File Uploader
if st.session_state.df is None:
    uploaded_file = st.file_uploader("Choose a CSV file.", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Form for Question
with st.form("Question"):
    question = st.text_input("Question", value="", type="default")
    submitted = st.form_submit_button("Submit")
    if submitted:
        try:
            with st.spinner():
                relevant_rows = find_relevant_rows(question, st.session_state.df)
                answer = ask_openai_chat_model(question, relevant_rows.to_string(header=False, index=False))
                st.write(answer)
                st.session_state.prompt_history.append(question)
        except Exception as e:
            st.error(f"Error processing question: {e}")

# Display current DataFrame and prompt history
if st.session_state.df is not None:
    st.subheader("Current dataframe:")
    st.write(st.session_state.df)

st.subheader("Prompt history:")
st.write(st.session_state.prompt_history)

# Clear button
if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None