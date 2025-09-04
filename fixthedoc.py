import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import PyPDF2
import re
from fuzzywuzzy import process

# Download required NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download()
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_input_text(uploaded_file, text_input):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    st.warning("No extractable text found in the PDF. It may contain images or be encrypted.")
                    return ""
                return text
            else:
                text = uploaded_file.read().decode("utf-8")
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception as e:
            st.error(f"Error reading file: {str(e)}. Ensure the file is a valid .txt or .pdf with extractable text.")
            return ""
    return text_input.strip()

def get_bert_embedding(text):
    if not text:
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()[0]

def doc_generation_feedback(input_text, topic, tone):
    if not input_text:
        return "No text provided!"
    
    text_embedding = get_bert_embedding(input_text)
    
    def pad_vector(vec):
        return np.pad(vec, (0, 768 - len(vec)), 'constant')
    
    doc_vectors = [
        {"label": "API Documentation", "vector": pad_vector([0.8, 0.5, 0.3])},
        {"label": "User Guide", "vector": pad_vector([0.6, 0.7, 0.2])},
        {"label": "Formal Technical", "vector": pad_vector([0.9, 0.4, 0.1])},
        {"label": "Informal Tutorial", "vector": pad_vector([0.5, 0.8, 0.3])}
    ]
    
    similarities = []
    for vec in doc_vectors:
        sim = cosine_similarity([text_embedding], [vec["vector"]])[0][0]
        similarities.append((vec["label"], sim))
    
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    feedback = f"Analysis for '{topic}' with '{tone}' tone:\n"
    for label, sim in ranked:
        feedback += f"- {label}: Similarity {sim:.2f}\n"
    if ranked[0][1] < 0.5:
        feedback += "\nSuggestion: Improve by adding more structure, precise terms, or examples."
    else:
        feedback += "\nYour text aligns well! Consider minor tweaks for better readability."
    
    return feedback

def intelligent_search(doc_text, query):
    if not doc_text or not query:
        return "No document or query provided!"
    
    sentences = sent_tokenize(doc_text)
    
    query_emb = get_bert_embedding(query)
    
    semantic_matches = []
    for sent in sentences:
        sent_emb = get_bert_embedding(sent)
        sim = cosine_similarity([query_emb], [sent_emb])[0][0]
        semantic_matches.append((sent, sim))
    semantic_ranked = sorted(semantic_matches, key=lambda x: x[1], reverse=True)[:5]
    
    fuzzy_matches = process.extract(query, sentences, limit=5)
    
    return {"semantic": semantic_ranked, "fuzzy": fuzzy_matches}

def doc_maintenance(old_text, new_text):
    if not old_text or not new_text:
        return "Provide both old and new texts!"
    
    old_emb = get_bert_embedding(old_text)
    new_emb = get_bert_embedding(new_text)
    sim = cosine_similarity([old_emb], [new_emb])[0][0]
    
    feedback = f"Similarity between old and new: {sim:.2f}\n"
    if sim < 0.7:
        feedback += "Significant changes detected. Suggest updating the documentation to reflect new content."
    else:
        feedback += "Minor changes. Documentation is mostly up-to-date."
    
    return feedback

st.title("Smart Documentation Assistant")

tab1, tab2, tab3 = st.tabs(["Check", "Search", "Compare"])

with tab1:
    st.header("Doc Creation Assistant")
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"], key="create_upload")
    input_text = st.text_area("Or enter your doc draft or code:", value="")
    if uploaded_file:
        extracted_text = get_input_text(uploaded_file, "")
        if extracted_text:
            st.write("Extracted Text Preview (first 500 characters):")
            st.text(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    topic = st.selectbox("Desired Topic", ["API Documentation", "User Guide"])
    tone = st.selectbox("Desired Tone", ["Formal Technical", "Informal Tutorial"])
    if st.button("Analyze & Get Feedback"):
        text_to_process = get_input_text(uploaded_file, input_text)
        if text_to_process:
            feedback = doc_generation_feedback(text_to_process, topic, tone)
            st.write(feedback)
        else:
            st.write("Please provide text via upload or input.")

with tab2:
    st.header("Intelligent Search in Docs")
    uploaded_doc = st.file_uploader("Upload documentation (.txt or .pdf)", type=["txt", "pdf"], key="consume_upload")
    doc_text = st.text_area("Or paste your documentation text:", value="")
    if uploaded_doc:
        extracted_doc = get_input_text(uploaded_doc, "")
        if extracted_doc:
            st.write("Extracted Document Preview (first 500 characters):")
            st.text(extracted_doc[:500] + "..." if len(extracted_doc) > 500 else extracted_doc)
    query = st.text_input("Search Query:", value="")
    if st.button("Search"):
        doc_to_process = get_input_text(uploaded_doc, doc_text)
        if doc_to_process:
            results = intelligent_search(doc_to_process, query)
            if isinstance(results, str):
                st.write(results)
            else:
                st.subheader("Semantic Matches (BERT-based):")
                for sent, score in results["semantic"]:
                    st.write(f"Score: {score:.2f} - {sent}")
                st.subheader("Fuzzy Keyword Matches:")
                for sent, score in results["fuzzy"]:
                    st.write(f"Score: {score} - {sent}")
        else:
            st.write("Please provide documentation via upload or input.")

with tab3:
    st.header("Doc Maintenance")
    uploaded_old = st.file_uploader("Upload old documentation (.txt or .pdf)", type=["txt", "pdf"], key="old_upload")
    old_text = st.text_area("Or enter old documentation:", value="")
    if uploaded_old:
        extracted_old = get_input_text(uploaded_old, "")
        if extracted_old:
            st.write("Extracted Old Document Preview (first 500 characters):")
            st.text(extracted_old[:500] + "..." if len(extracted_old) > 500 else extracted_old)
    uploaded_new = st.file_uploader("Upload new code or text (.txt or .pdf)", type=["txt", "pdf"], key="new_upload")
    new_text = st.text_area("Or enter new code or updated text:", value="")
    if uploaded_new:
        extracted_new = get_input_text(uploaded_new, "")
        if extracted_new:
            st.write("Extracted New Document Preview (first 500 characters):")
            st.text(extracted_new[:500] + "..." if len(extracted_new) > 500 else extracted_new)
    if st.button("Compare & Suggest"):
        old_to_process = get_input_text(uploaded_old, old_text)
        new_to_process = get_input_text(uploaded_new, new_text)
        if old_to_process and new_to_process:
            feedback = doc_maintenance(old_to_process, new_to_process)
            st.write(feedback)
        else:
            st.write("Please provide both old and new texts via upload or input.")