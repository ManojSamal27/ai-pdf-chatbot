import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Streamlit config
st.set_page_config(
    page_title="AI PDF Chatbot"
)

# App title
st.title("AI PDF Chatbot")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vector DB memory
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# Process PDF
if uploaded_file and st.session_state.vectorstore is None:

    os.makedirs("uploads", exist_ok=True)

    # Save uploaded PDF
    pdf_path = f"uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.success("PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="vectorstore"
    )

    # Save vectorstore in session
    st.session_state.vectorstore = vectorstore

    st.success("PDF processed successfully")

# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question
question = st.chat_input(
    "Ask question from PDF"
)

# Process question
if question and st.session_state.vectorstore:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Similarity search with score
    results = st.session_state.vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    context = ""

    sources = []

    retrieved_chunks = []

    # Process retrieved chunks
    for doc, score in results:

        context += doc.page_content + "\n"

        retrieved_chunks.append(
            {
                "content": doc.page_content,
                "score": score
            }
        )

        if "page" in doc.metadata:
            sources.append(
                f"Page {doc.metadata['page'] + 1}"
            )

    # Generate AI response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a PDF assistant.

                STRICT RULES:
                1. Answer ONLY using the provided PDF context.
                2. If answer is not present in context,
                   reply exactly:
                   "Answer not found in uploaded PDF."
                3. Do NOT use external knowledge.
                4. Do NOT make assumptions.
                5. Keep answers concise.

                PDF Context:
                {context}
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    # Extract answer
    answer = response.choices[0].message.content

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Display assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        # Display sources
        st.markdown("### Sources")

        for source in set(sources):
            st.markdown(f"- {source}")

        # Display retrieved chunks
        st.markdown("### Retrieved Chunks")

        for chunk in retrieved_chunks:

            st.markdown(
                f"""
                **Similarity Score:** {chunk['score']}

                {chunk['content']}
                """
            )
