import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# --- Page Config ---
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #DDDDDD;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #1F2228;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .user-msg {
            background-color: #0078FF;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 5px;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .bot-msg {
            background-color: #3B3E45;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 5px;
            max-width: 80%;
            float: left;
            clear: both;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>📄 Medical Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a PDF, and chat with it like a human!</p>", unsafe_allow_html=True)

# --- API Key ---
google_api_key = st.text_input("🔑 Enter your Google API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# --- PDF Upload ---
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

# --- Session State for Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Process PDF ---
if uploaded_file and google_api_key and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file to a temporary location
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF using PyPDFLoader (simpler, text-only)
        loader = PyMuPDFLoader("uploaded_file.pdf")
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)

        # Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)

        # Memory for conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        # --- Corrected Prompt Template ---
        # The prompt needs to be a single template with all variables
        template = """
        You are a helpful AI assistant. You are an expert on the uploaded PDF document.
        Answer the question based only on the following context, and the provided chat history if available.
        Do not use any external knowledge.
        Your response must always be in English.
        If the answer is not in the document, state that you do not have the information.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        QA_PROMPT = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])

        # --- Re-build your ConversationalRetrievalChain with the corrected prompt ---
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )

    st.success("✅ PDF processed! You can start chatting below.")

# --- Chat UI ---
if st.session_state.qa_chain:
    query = st.chat_input("💬 Ask something about the PDF...")
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({"question": query})
            answer = result["answer"]

        # Store in chat history
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", answer))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{message}</div>", unsafe_allow_html=True)

elif not google_api_key:
    st.warning("Please enter your Google API Key and upload a PDF to begin.")