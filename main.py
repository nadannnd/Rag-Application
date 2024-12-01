import os
import tempfile
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
import streamlit as st

# Utility function to load documents
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Choose loader based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        documents.extend(loader.load())
        os.unlink(temp_file_path)  # Clean up temporary file
    
    return documents

# Function to create vector store
def create_vector_store(documents, model_name):
    # Unique persist directory for each model
    persist_directory = f"C:\\intel\\Desktop\\chat project\\pdf_chroma_db_{model_name}"
    os.makedirs(persist_directory, exist_ok=True)

    # Create embeddings
    embed_model = OllamaEmbeddings(model=model_name, base_url='http://127.0.0.1:11434')

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    # Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embed_model,
        persist_directory=persist_directory
    )
    
    return vector_store, embed_model

# Main Streamlit App
def main():
    st.title("Multi-Model Document Q&A")

    # Model selection
    selected_model = st.selectbox(
        "Select LLM Model", 
        ["llama3.1", "mistral", "gemma:7b"]
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or Text Files", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True
    )

    # Session state to store vector store and model
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.embed_model = None

    # Vector store and embedding creation
    if uploaded_files:
        with st.spinner('Processing documents...'):
            # Load documents
            documents = load_documents(uploaded_files)
            
            # Create vector store
            st.session_state.vector_store, st.session_state.embed_model = create_vector_store(documents, selected_model)
            
            st.success(f'Processed {len(documents)} documents with {selected_model} model')

    # LLM and Retriever setup
    if st.session_state.vector_store:
        # Retriever
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

        # LLM
        llm = OllamaLLM(model=selected_model, base_url='http://127.0.0.1:11434')

        # Prompt Template
        prompt = ChatPromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Helpful Answer:""")

        # Create document chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Question input with larger width
            question = st.text_input(
                "Ask a question about your documents", 
                placeholder="Enter your question here..."
            )
        
        with col2:
            # Answer button aligned with input
            st.write("") # Add some spacing
            answer_button = st.button("Get Answer", type="primary")

        # Answer generation
        if answer_button and question:
            with st.spinner('Generating answer...'):
                try:
                    # Retrieve relevant documents
                    context_docs = retriever.invoke(question)
                    
                    # Prepare the input for the chain
                    chain_input = {
                        "context": context_docs,
                        "question": question
                    }
                    
                    # Invoke the chain
                    response = combine_docs_chain.invoke(chain_input)
                    
                    st.write("*Answer:*", response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Add more detailed error logging
                    st.error(f"Context docs: {context_docs}")
                    st.error(f"Chain input: {chain_input}")

    else:
        st.info("Please upload documents to start")

# Run the app
if __name__ == "__main__":
    main()