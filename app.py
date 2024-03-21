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
import requests
from io import BytesIO

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

def getPDFText(PDFDocs):
    text=""
    for pdf in PDFDocs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def urlPDF(pdfURL):
    response = requests.get(pdfURL)
    pdfFile = BytesIO(response.content)
    return pdfFile




def getTextChunks(text):
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = textSplitter.split_text(text)
    return chunks


def getVectorStore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def getConvoChain():

    prompt_template = """
    You are an AI named Norman. Norman is quite formal but can also be a wisecracking fellow. Answer like Norman.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. 
    \n\n
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
    
    newDB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = newDB.similarity_search(user_question)

    chain = getConvoChain()

    
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

    if 'chatHistory' not in st.session_state:
        st.session_state['chatHistory'] = []

    for role,text in st.session_state['chatHistory']:
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
        pdfDocs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        pdfURL = st.text_input("Enter PDF URL")

        st.session_state.pdfMode = True
        with st.spinner("Processing..."):
            if st.button("Submit & Process"):

                    if pdfDocs and pdfURL:
                        st.warning("Please remove either the file or the URL as Norman can only process one at a time.")
                    elif pdfDocs:
                            raw_text = getPDFText(pdfDocs)
                    elif pdfURL:
                            pdf_file = urlPDF(pdfURL)
                            raw_text = getPDFText([pdf_file])
                    else:
                            st.error("Please upload PDFs or provide a PDF URL.")
                            return
                
                    text_chunks = getTextChunks(raw_text)
                    getVectorStore(text_chunks)
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

        st.session_state['chatHistory'].append(("You", userQuestion))
        st.session_state['chatHistory'].append(("Bot", response))
    



if __name__ == "__main__":
    main()