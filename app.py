from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
import random
import time

def streamSim(sentence):
    for word in sentence.split():
        yield word + " "
        time.sleep(0.05)

llm = ChatGoogleGenerativeAI(model="gemini-pro")

def ask(question):
    result = llm.invoke(question)
    return result.content

st.set_page_config(page_title="Shaikh's ChatBot")

st.header("Gemini LLM App")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for role,text in st.session_state['chat_history']:
    with st.chat_message(role):
        st.markdown(text)

if input := st.chat_input("What would you like to ask me?", key=input):
    with st.chat_message("You"):
        st.markdown(input)

if input:
    response = ask(input)
    with st.chat_message("Bot"):
        st.write_stream(streamSim(response))

    st.session_state['chat_history'].append(("You", input))
    st.session_state['chat_history'].append(("Bot", response))







