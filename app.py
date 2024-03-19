from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os

llm = ChatGoogleGenerativeAI(model="gemini-pro")

def ask(question):
    result = llm.invoke(question)
    return result.content

st.set_page_config(page_title="Shaikh's ChatBot")

st.header("Gemini LLM App")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("What would you like to ask me? :", key=input)
submit = st.button("Ask")

if submit and input:
    response = ask(input)
    #
    st.session_state['chat_history'].append(("You", input))
    st.subheader("Response: ")
    st.write(response)
    st.session_state['chat_history'].append(("Bot", response))

st.subheader("Chat History")

for role,text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")



