import streamlit as st
import replicate
import os
import fitz  # PyMuPDF for PDF processing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot - Ice Skating")

# Sidebar for API Key and Model Selection
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')

    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and Parameters')
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'])
    llm = ('a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea' 
           if selected_model == 'Llama2-7B' 
           else 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5')

    temperature = st.slider('Temperature', 0.01, 5.0, 0.1, 0.01)
    top_p = st.slider('Top-p', 0.01, 1.0, 0.9, 0.01)
    max_length = st.slider('Max Length', 32, 128, 120, 8)

# Store messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with Ice Skating competitions?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with Ice Skating competitions?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Load PDFs and extract text
def load_pdfs(uploaded_files):
    documents = []
    for file in uploaded_files:
        try:
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                if text.strip():
                    documents.append(text)
                else:
                    st.warning(f"⚠️ No text extracted from {file.name}.")
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    if documents:
        st.success(f"✅ Successfully loaded {len(documents)} document(s).")
    return documents

# Create FAISS index from text embeddings
def create_vector_database(texts):
    if not texts:
        st.error("❌ No text data found to create embeddings.")
        return None, None  # Avoid creating an empty index

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.success(f"✅ Created FAISS index with {len(texts)} document(s).")
    return index, texts

# Sidebar for PDF Upload
st.sidebar.subheader("Upload PDFs for Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Load PDFs and create FAISS index only when new files are uploaded
if uploaded_files:
    pdf_texts = load_pdfs(uploaded_files)
    vector_index, doc_texts = create_vector_database(pdf_texts)
    st.session_state.vector_index = vector_index
    st.session_state.doc_texts = doc_texts
else:
    vector_index = st.session_state.get("vector_index", None)
    doc_texts = st.session_state.get("doc_texts", None)

# Retrieve relevant knowledge from FAISS
def retrieve_context(query, k=3, threshold=1.0, max_chars=1500):
    """Retrieve relevant knowledge but limit context size."""
    if vector_index is None or doc_texts is None:
        st.warning("⚠️ No knowledge base available. Please upload PDFs.")
        return None

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = vector_index.search(query_embedding, k=k)

    relevant_texts = [doc_texts[i] for i, dist in zip(I[0], D[0]) if dist < threshold]
    context_text = "\n\n".join(relevant_texts)

    if relevant_texts:
        st.success(f"✅ Retrieved {len(relevant_texts)} relevant document(s) for query.")
    else:
        st.warning("⚠️ No relevant information found for the query.")

    return context_text[:max_chars] + "..." if len(context_text) > max_chars else context_text

# LLaMA-2 Response Generation with Context
def generate_llama2_response(prompt_input, max_input_length=3500):
    """Generate a response ensuring prompt does not exceed token limit"""
    context = retrieve_context(prompt_input)
    
    # Format prompt with retrieved context
    if context:
        prompt_context = f"Use the following relevant information to answer the query accurately:\n\n{context}\n\n### User Query ###\n{prompt_input}"
    else:
        prompt_context = prompt_input  # If no relevant context is found
    
    # Ensure prompt length is within Llama 2 limits
    if len(prompt_context) > max_input_length:
        prompt_context = prompt_context[:max_input_length] + "..."
    
    # Run LLaMA-2 with improved prompt
    output = replicate.run(llm, input={"prompt": prompt_context, "temperature": temperature, "top_p": top_p, "max_length": max_length})
    
    return "".join(output)

# User input
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate response if needed
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
