import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI


# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None if extract_text fails
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, model_choice):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,max_memory=4
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a chatbot that answers questions based on provided documents."),
            ("human", "Question: {question}")
        ]
    )

    if model_choice == "OpenAI":
        llm = ChatOpenAI()
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=150,  # Limit max tokens for the response
            timeout=None,
            max_retries=2,
        )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message.content}")
            else:
                st.write(f"**Bot:** {message.content}")
    else:
        st.write("Please upload and process PDF files first.")

def main():
    st.set_page_config(page_title="Chat with PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""

    st.header("Chat with PDF ")

    model_choice = st.selectbox("Select AI Model:", ["OpenAI", "Gemini"])
    user_question = st.text_input("Ask a question about your documents:")

    # Send button to submit the question
    if st.button("Send"):
        if user_question:
            handle_user_input(user_question)
        else:
            st.warning("Please enter a question before sending.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.extracted_text = raw_text  # Store extracted text

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, model_choice)

    # Display extracted text
    if st.session_state.extracted_text:
        st.subheader("Extracted Text from PDFs")
        st.text_area("Extracted Text:", value=st.session_state.extracted_text, height=300)

if __name__ == '__main__':
    main()
