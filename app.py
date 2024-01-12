# For UI
import streamlit as st
# load Hugging_face_api key
from dotenv import load_dotenv
# Modify PDF 
from PyPDF2 import PdfReader
# real-time data processing and integration with Large Language Models
from langchain.text_splitter import CharacterTextSplitter
# Hugging Face is a company that specializes in natural language processing (NLP) and
# provides a wide range of pre-trained models and tools for NLP tasks.
from langchain.embeddings import HuggingFaceInstructEmbeddings
# Faiss is an open-source library for fast, dense vector similarity search and grouping
# It is commonly used for tasks such as nearest neighbor search and clustering.
from langchain.vectorstores import FAISS
# Loading large language model from langchain
from langchain.llms import HuggingFaceHub
# Temporary memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# Importing css styles from html_temp file
from html_temp import css, bot_template, user_template


def main():
    # hugging face apikey
    load_dotenv()
    # build the UI
    st.set_page_config(page_title="Pdf-Bot",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>PDF-BOT📚</h1>", unsafe_allow_html=True)

    # initiate the session_state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # file uploader from the
    pdf_docs = st.file_uploader(
        "Upload your PDF's here and click the 'Process' button!", 
        accept_multiple_files=True,
        )
    
    # button
    if st.button("Process"):
        with st.spinner("Reading"):
            # get PDFs texts
            raw_text = get_pdfs_texts(pdf_docs)
            # get the texts chunks
            text_chunks = get_text_chunks(raw_text)
            # create vector store
            vectorstore = get_vectorestore(text_chunks)
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    # user question Box
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        handle_userinput(user_question)

def get_pdfs_texts(pdf_docs):
    texts = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()
    return texts

def get_text_chunks(text):
    # Splits the extracted text into smaller chunks using the CharacterTextSplitter class.
    # Character-Based Splitting,Flexible Chunk Size Management,Customizable Separators,Optional Chunk Overlap,Metadata Preservation,
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Creating embeddings for the text chunks using a Hugging Face model (hkunlp/instructor-xl).
# The Hugging Face model hkunlp/instructor-xl is likely being used in this code for generating embeddings
# (vector representations) of the text chunks extracted from the PDF documents
def get_vectorestore(chunks):
    # first converting the text into a sequence of tokens.
    # These tokens are then used to generate a vector representation of the text
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")#model_kwargs={"device": "cuda:0"}
    # FAISS.from_texts(): A function from FAISS designed for building vector stores from text data.
    # assures faster similarity searching when the number of vectors may go up to millions or billions.
    # is a powerful tool for efficient similarity search and clustering of high-dimensional data,
    # enabling developers to quickly find similar items in large datasets.
    # Some practical applications of FAISS include image retrieval, recommendation systems, and natural language processing
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.9, "max_length":512, "device": "cuda:0"})
    # Temparature==>  If you prefer more focused and deterministic output, you might use a lower temperature.
    # max_length==>the length of the generated sequence will be limited to a maximum of 512 tokens.
    
    #llama have more than 32 transformer layers but the flan t5 have 12 transformer layers 
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.9, "max_length":512})
    # This memory allows for storing messages and then extracts the messages in a variable.
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # a specialized chain type from LangChain designed for conversational tasks.
    # vectorstore.as_retriever():
    # This part converts the provided vectorstore into a compatible format for use as a retriever within the chain.
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()