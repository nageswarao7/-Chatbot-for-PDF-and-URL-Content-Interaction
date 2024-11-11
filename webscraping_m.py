import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
import os

# Load environment variables
load_dotenv()

def get_web_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        st.error("Failed to retrieve the webpage.")
        return ""

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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, model_choice):
    if model_choice == "OpenAI":
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model='gpt-4o-2024-08-06')
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=150,
            timeout=None,
            max_retries=2,
        )

    # Using RetrievalQA instead of ConversationalRetrievalChain
    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # or "map_reduce" based on your needs
        retriever=vectorstore.as_retriever()
    )
    
    return conversation_chain

def handle_user_input(user_question, vectorstore, model_choice):
    conversation = get_conversation_chain(vectorstore, model_choice)
    response = conversation({"query": user_question})
    return response['result']

def main():
    st.set_page_config(page_title="Chat with Web Pages")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with Web Pages")

    model_choice = st.selectbox("Select AI Model:", ["OpenAI", "Gemini"])
    user_question = st.text_input("Ask a question about the webpage content:")

    # Send button to submit the question
    if st.button("Send"):
        if user_question:
            if st.session_state.vectorstore is not None:
                answer = handle_user_input(user_question, st.session_state.vectorstore, model_choice)
                st.session_state.chat_history.append({"question": user_question, "answer": answer})

                for chat in st.session_state.chat_history:
                    st.write(f"**You:** {chat['question']}")
                    st.write(f"**Bot:** {chat['answer']}")
            else:
                st.write("Please enter a URL and process the content first.")
        else:
            st.warning("Please enter a question before sending.")

    with st.sidebar:
        st.subheader("Webpage URLs")
        url1 = st.text_input("Enter the first URL of the webpage:")
        url2 = st.text_input("Enter the second URL of the webpage:")
        url3 = st.text_input("Enter the third URL of the webpage:")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                all_text = ""
                
                for url in [url1, url2, url3]:
                    if url:  # Only process non-empty URLs
                        all_text += get_web_text(url) + "\n"
                
                st.session_state.extracted_text = all_text.strip()  # Store combined extracted text

                # Get the text chunks
                text_chunks = get_text_chunks(all_text)

                # Create vector store
                st.session_state.vectorstore = get_vectorstore(text_chunks)  # Store vectorstore in session state

    # Display extracted text
    if st.session_state.extracted_text:
        st.subheader("Extracted Text from Webpages")
        st.text_area("Extracted Text:", value=st.session_state.extracted_text, height=300)

if __name__ == '__main__':
    main()
