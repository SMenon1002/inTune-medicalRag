import streamlit as st
from google import genai
from google.genai import types
from rag_prep import RAGPreprocessor
from config import (
    GEMINI_API_KEY, 
    GEMINI_CHAT_MODEL, 
    DEFAULT_RESULTS_COUNT,
    logger
)
from exceptions import ModelGenerationError

client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize RAG preprocessor
@st.cache_resource
def initialize_rag():
    logger.info("Initializing RAG from persistent storage")
    return RAGPreprocessor(google_api_key=GEMINI_API_KEY)

# Initialize the chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def get_relevant_context(query: str, rag_prep: RAGPreprocessor) -> str:
    """Get relevant context from RAG system for the given query."""
    results = rag_prep.collection.query(
        query_texts=[query],
        n_results=DEFAULT_RESULTS_COUNT
    )
    
    context_parts = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        context_parts.append(
            f"\nSource: {metadata['source']}, Page: {metadata['page']}\n{doc}\n"
        )
    
    return "".join(context_parts)

def generate_response(prompt: str, context: str) -> str:
    """Generate a response using the Gemini model."""
    formatted_prompt = f"""**System instructions:**
You are a helpful medical assistant. Your role is to provide accurate medical information based on authoritative sources.
You must:
- Only use information from the provided context
- Say when you cannot find relevant information
- Use clear, professional medical terminology while remaining accessible
- Never make up or infer information not present in the context
- Focus on accuracy and factual information

**Context:**
{context}

**User query:**
{prompt}"""
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_prompt)],
        )
    ]
    
    generate_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    
    try:
        response_stream = client.models.generate_content_stream(
            model=GEMINI_CHAT_MODEL,
            contents=contents,
            config=generate_config,
        )
        
        return "".join(chunk.text for chunk in response_stream if chunk.text)
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        raise ModelGenerationError(error_msg)

def main():
    # Set up the Streamlit page
    st.set_page_config(page_title="Medical Chat Assistant", page_icon="🏥")
    st.title("Medical Chat Assistant 🏥")

    try:
        # Initialize RAG
        rag_prep = initialize_rag()
        
        # Display chat messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input
        if prompt := st.chat_input("What would you like to know about hypertension?"):
            # Add user message to chat history
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Get relevant context from RAG
                context = get_relevant_context(prompt, rag_prep)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_response(prompt, context)
                        st.markdown(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
            
            except ModelGenerationError as e:
                st.error("Sorry, I'm having trouble generating a response. Please try again.")
                logger.error(f"Model generation error: {str(e)}")
            except Exception as e:
                st.error("An unexpected error occurred. Please try again.")
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)

    except Exception as e:
        st.error("Failed to initialize the chat assistant. Please try again later.")
        logger.error(f"Application initialization error: {str(e)}", exc_info=True)

    # Add a sidebar with information
    with st.sidebar:
        st.markdown("""
        ### About
        This chat assistant uses:
        - Gemini AI
        - RAG (Retrieval Augmented Generation)
        - Medical guidelines on hypertension
        
        Ask questions about hypertension diagnosis, treatment, and management.
        """)

if __name__ == "__main__":
    main() 