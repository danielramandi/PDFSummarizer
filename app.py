import openai
import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI


def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type, 
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    return summaries


@st.cache_data
def setup_documents(uploaded_file, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        loader = PyPDFLoader(tmp.name)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs


def main():
    st.set_page_config(layout="wide")
    st.title("Custom Summarization App")
    
    openai.api_key = st.sidebar.text_input("OpenAI API Key")

    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=10000, step=100, value=1900)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=10000, step=100, value=200)
    user_prompt = st.text_input("Enter the user prompt")
    uploaded_file = st.file_uploader("Upload a pdf file", type=['pdf'])
    temperature = st.sidebar.number_input("ChatGPT Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
    num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)
    
    llm = st.sidebar.selectbox("LLM", ["ChatGPT", "GPT4", ""])
    if llm == "ChatGPT":
        llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=temperature)
    elif llm == "GPT4":
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai.api_key, temperature=temperature)
    
    if uploaded_file is not None:
        docs = setup_documents(uploaded_file, chunk_size, chunk_overlap)
        st.write("PDF was loaded successfully")
        
        if st.button("Summarize"):
            result = custom_summary(docs, llm, user_prompt, chain_type, num_summaries)
            st.write("Summaries:")
            for summary in result:
                st.write(summary)

if __name__ == "__main__":
    main()
