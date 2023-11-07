import streamlit as st
import os
from dotenv import load_dotenv
import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import  ConversationalRetrievalChain
load_dotenv()
print("Streamlit run has started")
# Title
st.title("Ask PDF")


target_dir = "./uploaded_data"
try :
    for file in os.listdir(target_dir):
        os.remove(os.path.join(target_dir + "/" + file))
except :
    print("Error in accessing the target directory ")

# Uploading Files
uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)


@st.cache_resource
def load_uploaded_files(uploaded_files: list,target_dir:str):
    if len(uploaded_files) > 0:
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"length of the splits {len(splits)}")
        embedding = OpenAIEmbeddings()
        print("loaded the Open AI Embeddings Function")

        # Deleting the previous content
        # shutil.rmtree("./docs/chroma/")
        # print("Deleted the db")

        persist_directory = 'docs/chroma/'
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        retriever = vectordb.as_retriever()

        print("Created Vector DB ")
        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            llm_name = "gpt-3.5-turbo-0301"
        else:
            llm_name = "gpt-3.5-turbo"
        print(llm_name)
        print("chose the llm")

        return llm_name, retriever

if len(uploaded_files) > 0 :
    llm_name,retriever = load_uploaded_files(uploaded_files,target_dir=target_dir)

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
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



