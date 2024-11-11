# Chat with Web Pages and PDFs

This Streamlit app allows users to interact with web pages and PDF documents via a conversational AI. It leverages web scraping to extract text from webpages and PDF files, stores the extracted content in a vector store, and enables users to ask questions about the content. The responses are powered by language models like OpenAI GPT and Google's Gemini.

## Features
- **Web Scraping**: Extracts and processes text from URLs provided by the user.
- **PDF Text Extraction**: Upload multiple PDF files and extract the text.
- **Conversational AI**: Ask questions about the extracted content using models like OpenAI's GPT or Google's Gemini.
- **Vector Store**: Uses FAISS to store text embeddings for fast and accurate retrieval.
- **Memory**: Keeps track of previous chat history to simulate a more interactive experience.
