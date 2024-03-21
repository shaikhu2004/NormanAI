import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


llm=ChatGoogleGenerativeAI(model="gemini-pro")


def ask(question):
    fullQuestion = "You are an AI named Norman. Norman is quite formal but can also be a wisecracking fellow. Answer like Norman. \n"+question
    result = llm.invoke(fullQuestion)
    return result.content

def streamSim(sentence):
    for word in sentence.split():
        yield word + " "
        time.sleep(0.05)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. 
    You are an AI named Norman. Norman is quite formal but can also be a wisecracking fellow. Answer like Norman.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def askPDF(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)

    return(response["output_text"])






def main():
    st.set_page_config("Norman AI")
    st.header("Chat with Norman 🤖")
    st.divider()

    userPNG = "https://static.vecteezy.com/system/resources/thumbnails/006/090/662/small_2x/user-icon-or-logo-isolated-sign-symbol-illustration-free-vector.jpg"
    botPNG = "https://img.freepik.com/premium-vector/simple-robot-line-illustration_168578-329.jpg"

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for role,text in st.session_state['chat_history']:
        if role=="You":
            pfp=userPNG
        else:
            pfp=botPNG
        with st.chat_message(role, avatar=pfp):
            st.markdown(text)

    if "pdfMode" not in st.session_state:
        st.session_state.pdfMode = False

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            st.session_state.pdfMode = True
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                st.markdown("Next Question asked will be answered according to uploaded PDF.")

    if userQuestion := st.chat_input("Ask a Question:"):
        with st.chat_message("You", avatar=userPNG):
            st.markdown(userQuestion)



    if userQuestion:
        if st.session_state.pdfMode:
            try:
                response=askPDF(userQuestion)
            except Exception:
                response=ask("Say one line in case the program crashes")
            st.session_state.pdfMode=False
        else:
            try:
                response=ask(userQuestion)
            except Exception:
                response=ask("Say one line in case the program crashes")
        with st.chat_message("Bot", avatar=botPNG):
            st.write_stream(streamSim(response))

        st.session_state['chat_history'].append(("You", userQuestion))
        st.session_state['chat_history'].append(("Bot", response))
    



if __name__ == "__main__":
    main()