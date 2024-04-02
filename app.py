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
    - Please provide thorough and structured answers, avoiding shortcuts and ensuring accuracy.
- Format your responses in markdown for better readability.
- When explaining steps or processes, provide detailed instructions.
- Always clearly indicate the user prompt for each question.
- Generate responses without including labels such as 'User Prompt:' and 'Bot Response:'.
- Avoid duplicating the user prompt in your response.
- Sometimes you will be asked to respond to prompts that may not be in the form of a question or order; just respond accordingly.
- If the prompt contains non-English characters or words, just respond that you can't understand it.

Please keep these guidelines in mind when generating your responses, and make sure to provide steps if you used any.

Your context and questions are as follows:
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)

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
        st.title("PDF Menu:")
        if st.session_state.pdfMode==True:
                st.markdown("PDF Mode is ON.")
        else:
                st.markdown("PDF Mode is OFF.")
        pdfDocs = st.file_uploader("Upload your PDF Files and Click on the Enter PDF Mode Button", accept_multiple_files=True)
        pdfURL = st.text_input("Enter PDF URL")

        
        with st.spinner("Processing..."):

            if st.button("Enter PDF Mode"):
                    st.session_state.pdfMode = True

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
                    st.markdown("You are now in PDF Mode. All Questions asked will only be answered according to the uploaded PDF.")

        if st.session_state.pdfMode==True:
            if st.button("Stop PDF Mode"):
                st.session_state.pdfMode = False
            

    if userQuestion := st.chat_input("Ask a Question:"):
        with st.chat_message("You", avatar=userPNG):
            st.markdown(userQuestion)



    if userQuestion:
        if st.session_state.pdfMode:
            try:
                response=askPDF(userQuestion)
            except Exception:
                response=ask("Say one line in case the program crashes")
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