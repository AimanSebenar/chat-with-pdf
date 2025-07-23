import streamlit as st
from utils import extract_text_pdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOllama

st.set_page_config(page_title="Chat with PDF", layout = "centered")
st.title("Chat with PDF via Llama3")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    text = extract_text_pdf(uploaded_file)
    st.text_area("Extracted text",text) #debug

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200, 
                                               separators=["\n\n", "\n", ".", "!", "?", " "])
    chunks = splitter.split_text(text)
    st.write(f"Num of chunks: {len(chunks)}") #debug

    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="llama3")

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about the PDF (please refer to the file as \"text\"): ")

    if query:
        docs = retriever.get_relevant_documents(query)
        st.write("Retrieved docs preview:",[d.page_content[:300] for d in docs])
        with st.spinner("Thinking..."):
            answer = qa.run(query)
            st.success(answer)