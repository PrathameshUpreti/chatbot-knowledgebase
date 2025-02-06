import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")
CSV_PATH = os.getenv("CSV_PATH", "Position_Salaries.csv")

def highlight_credential(key_name, key_value):
    return f"**{key_name}:** `{key_value if key_value else 'Not Provided'}`"
def load_csv_to_vectorstore(csv_path, embedding):
    chunksize = 10000 
    documents = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            doc = Document(
                page_content=str(row.to_dict()),  
                metadata={"source": csv_path} 
            )
            documents.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embedding)
st.set_page_config(page_title="ChatBot Application", page_icon='🤖')
st.title("ChatBot using LangChain & LLaMA")

st.markdown(highlight_credential("Hugging Face API Key", HUGGINGFACE_API_KEY))
st.markdown(highlight_credential("CSV Path", CSV_PATH))

huggingface_api_key = st.text_input("Enter your API key:", type="password", value=HUGGINGFACE_API_KEY or "")
if huggingface_api_key:
    repo_id = "meta-llama/Llama-3.2-1B"
    llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation", max_length=150, temperature=0.7, token=huggingface_api_key)

    session_id = st.text_input("Session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = load_csv_to_vectorstore(CSV_PATH, embedding)
    retriever = db.as_retriever()
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it as a standalone question if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use only the provided CSV data as context. Do not provide any answers outside of the CSV file. "
         "Include inline citations in the format `**Source: {{metadata[source]}}**`. "
         "If the answer is not available in the CSV, state that explicitly. "
         "Do not provide outside context."
         "\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Function to get session history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the API Key or set it in the .env file.")
