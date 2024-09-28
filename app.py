import streamlit as st
from constants import EMBEDDING_MODEL
from embed_articles import load_documents
from embedding_utils import get_batched_embeddings
from get_ai_response import get_ai_response, MODEL
import numpy as np
import re
import html

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_relevant_chunks(query_embedding, documents, top_k=5):
    all_chunks = [chunk for doc in documents for chunk in doc.chunks]
    similarities = [cosine_similarity(query_embedding, np.array(chunk.embedding)) for chunk in all_chunks]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [all_chunks[i] for i in top_indices]

# Define system prompts
SYSTEM_PROMPTS = {
    "Default": """You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain relevant information, you can draw from your general knowledge. When using information from the context, cite your sources using square brackets with a number, like this: [1]. Use a new number for each unique source.""",
    "Concise": """You are a concise AI assistant. Provide brief, to-the-point answers based on the given context. If the context is insufficient, use your knowledge but keep responses short. Always cite your sources using square brackets with a number, like this: [1]. Use a new number for each unique source.""",
    "Expert": """You are an expert AI assistant with deep knowledge in various fields. Provide detailed, nuanced answers using the context provided and your extensive expertise. Explain complex concepts clearly. When referencing information from the context, always include a citation using square brackets with a number, like this: [1]. Use a new number for each unique source.""",
    "Creative": """You are a creative AI assistant. Use the context as inspiration to provide imaginative and original answers. Think outside the box and offer unique perspectives on the user's questions. When drawing inspiration or information from the context, cite your sources using square brackets with a number, like this: [1]. Use a new number for each unique source.""",
}

def display_message_with_citations(message, context):
    # Split the message into parts based on citations
    parts = re.split(r'(\[\d+\])', message)
    
    html_message = ""
    for part in parts:
        if re.match(r'\[\d+\]', part):
            citation_number = int(part[1:-1])
            citation_text = html.escape(context[citation_number - 1].replace("\n", ""))
            html_message += f'<span class="citation" title="{citation_text}">{part}</span>'
        else:
            html_message += html.escape(part)
    
    st.markdown(
        f"""
        <div class="message-with-citations">
            {html_message}
        </div>
        <style>
            .message-with-citations {{
                font-size: 16px;
                line-height: 1.5;
            }}
            .citation {{
                color: #0645AD;
                cursor: pointer;
                text-decoration: underline;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Load embedded documents
    if "documents" not in st.session_state:
        st.session_state.documents = load_documents()
    documents = st.session_state.documents

    st.title("McCulloch Chat App")

    # Sidebar
    st.sidebar.title("Settings")
    selected_prompt_name = st.sidebar.selectbox("Select System Prompt", list(SYSTEM_PROMPTS.keys()))
    model = st.sidebar.selectbox("Select Model", ["sonnet", "haiku"])
    top_k = st.sidebar.slider("Number of relevant chunks", min_value=1, max_value=20, value=5)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    show_context = st.sidebar.checkbox("Show full context in chat", value=False)
    show_cost = st.sidebar.checkbox("Show API call cost", value=True)

    # System prompt editor
    st.subheader("System Prompt")
    system_prompt = st.text_area("Edit system prompt", value=SYSTEM_PROMPTS[selected_prompt_name], height=100)

    # Initialize chat history and total cost
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    # Display total cost
    if show_cost:
        st.sidebar.markdown(f"**Total Cost**: ${st.session_state.total_cost:.6f}")

    # Chat interface
    st.subheader("Chat")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_message_with_citations(message["content"], message["context"])
            else:
                st.markdown(message["content"])
            if show_context and "context" in message:
                with st.expander("Show full context"):
                    st.write(message["context"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        query_embedding = get_batched_embeddings([prompt], EMBEDDING_MODEL)[0]
        relevant_chunks = get_relevant_chunks(query_embedding, documents, top_k)
        context = [chunk.text for chunk in relevant_chunks]

        # Prepare human message with context
        human_message = f"Context:\n" + "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)]) + f"\n\nQuestion: {prompt}"

        st.session_state.messages.append({"role": "user", "content": prompt, "context": context})
        with st.chat_message("user"):
            st.markdown(prompt)
            if show_context:
                with st.expander("Show full context"):
                    st.write(context)

        conversation_history = [
            msg["content"] if msg["role"] == "assistant" else 
            f"Context:\n" + "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(msg['context'])]) + f"\n\nQuestion: {msg['content']}"
            for msg in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, cost = get_ai_response(system_prompt, conversation_history, model, temperature)
                display_message_with_citations(response, context)
                st.session_state.messages.append({"role": "assistant", "content": response, "context": context})
                st.session_state.total_cost += cost
            
            if show_cost:
                st.caption(f"Cost: ${cost:.6f}")

if __name__ == "__main__":
    main()