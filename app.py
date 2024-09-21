import streamlit as st
from constants import EMBEDDING_MODEL
from embed_articles import load_documents
from embedding_utils import get_batched_embeddings
from get_ai_response import get_ai_response, MODEL
import numpy as np

# Load embedded documents
documents = load_documents()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_relevant_chunks(query_embedding, top_k=5):
    all_chunks = [chunk for doc in documents for chunk in doc.chunks]
    similarities = [cosine_similarity(query_embedding, np.array(chunk.embedding)) for chunk in all_chunks]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [all_chunks[i] for i in top_indices]

def main():
    st.title("RAG-powered Chat App")

    # Sidebar
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Select Model", ["sonnet", "haiku"])
    top_k = st.sidebar.slider("Number of relevant chunks", min_value=1, max_value=10, value=5)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    show_context = st.sidebar.checkbox("Show context in chat", value=False)
    show_cost = st.sidebar.checkbox("Show API call cost", value=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant chunks
        query_embedding = get_batched_embeddings([prompt], EMBEDDING_MODEL)[0]
        relevant_chunks = get_relevant_chunks(query_embedding, top_k)
        context = "\n".join([chunk.text for chunk in relevant_chunks])

        # Prepare system message with context
        system_message = f"""You are a helpful AI assistant. Use the following context to answer the user's question:

        {context}

        If the context doesn't contain relevant information, you can draw from your general knowledge."""

        # Prepare conversation history
        conversation_history = [msg["content"] for msg in st.session_state.messages]

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, cost = get_ai_response(system_message, conversation_history, model, temperature)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            if show_cost:
                st.caption(f"Cost: ${cost:.6f}")
            
            if show_context:
                with st.expander("Show context"):
                    st.write(context)

if __name__ == "__main__":
    main()

