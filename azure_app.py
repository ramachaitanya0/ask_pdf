import shutil

import streamlit as st
import os
from dotenv import load_dotenv
import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains import  ConversationalRetrievalChain
from langchain import PromptTemplate
load_dotenv("./azure.env")
print("Streamlit run has started")
# Title
st.title("Ask PDF")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting file {file_path}: {e}")

def delete_folder(folder_path):
    try:
        os.rmdir(folder_path)
    except Exception as e:
        st.error(f"Error deleting folder {folder_path}: {e}")


TARGET_DIR = "./uploaded_data"

# Uploading Files
uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)

@st.cache_resource
def load_uploaded_files(uploaded_files: list,target_dir:str):
    # delete_files_in_folder("./docs")
    # delete_folder("./docs")
    # shutil.rmtree("./docs")
    # try :
    #     os.remove("./docs/chroma/chroma.sqlite3")
    # except :
    #     print("No such directory")
    if len(uploaded_files) > 0:
        # creates new folder for the uploaded data
        create_folder(target_dir)
        print("Created the directory for the uploaded files")

        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
        print("Written all the files successfully")
        st.write("Successfully Uploaded the files")

        # Loading pdfs
        docs = []
        for file  in os.listdir(target_dir):
            loader = PyPDFLoader(target_dir + '/' + file)
            docs.extend(loader.load())
        print(f"length of the docs {len(docs)}")


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,separators=["\n\n","\n"])
        splits = text_splitter.split_documents(docs)

        print(f"length of the splits {len(splits)}")
        embedding = OpenAIEmbeddings(deployment="ada-002")
        print("loaded the Open AI Embeddings Function")


        # Creating Vector Database and persisting it
        persist_directory = './docs/chroma/'
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        # Creating a retriever
        retriever = vectordb.as_retriever()
        print("Created Vector DB ")

        # Selecting a Model
        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            llm_name = "gpt-3.5-turbo-0301"
        else:
            llm_name = "gpt-3.5-turbo"
        print(llm_name)
        print("chose the llm")

        delete_files_in_folder(target_dir)
        delete_folder(target_dir)
        print("Deleted the files and directory")


        return llm_name, retriever


if len(uploaded_files) > 0 :
    PROMPT_TEMPLATE = """You are an AI Assistant. Please refer to the context given and answer your query.

Context: {context}
Question: {question}

As you respond to your query, adhere to these ground rules delimited by four hashtags:
####
    Begin the conversation with a warm greetings with the user.
    Maintain a consistently polite tone in all interactions.
    Provide answers solely relevant to the context , If the question falls outside this scope, politely mention that it's beyond your knowledge.
Comprehend the context thoroughly and offer responses based solely on the provided information. Refrain from generating irrelevant or speculative answers.
#### """
    input_variables = ['context','question']
    prompt = PromptTemplate(template=PROMPT_TEMPLATE,input_variables=input_variables)
    llm_name,retriever = load_uploaded_files(uploaded_files,target_dir=TARGET_DIR)

    # llm = AzureChatOpenAI(model_name=llm_name, temperature=0)
    llm = ChatOpenAI(model="gpt-4-32k", deployment_id="dse-copilot-gpt4-32k")
    chat_history = []
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                                     chain_type="stuff")


print("started Chatbot")
st.title("QA Bot")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "Assistant", "content": "How can i help you"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a Question")
#
if question is not None:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    result = qa_chain({"question": question, "chat_history": chat_history})['answer']
    chat_history = [(question, result)]
    st.session_state.messages.append({"role": 'Assistant', "content": result})
    st.chat_message("Assistant").write(result)




