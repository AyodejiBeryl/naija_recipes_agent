"""
Naija Recipes Agent - Nigerian Indigenous Food & Soup Recipe Assistant
A RAG-based AI agent that provides authentic Nigerian recipes from trusted,
published cookbook sources only. Refuses to fabricate recipes.
"""

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "naija_recipes"

SYSTEM_PROMPT = """You are the **Naija Recipes Agent** — an expert assistant specializing EXCLUSIVELY
in indigenous Nigerian soups, stews, and traditional food recipes.

YOUR MISSION:
You help users discover and cook authentic Nigerian dishes using ONLY information retrieved from
trusted, published Nigerian cookbooks in your knowledge base.

STRICT RULES:
1. **ONLY answer from the retrieved context below.** If the context does not contain information
   about the requested dish, say: "I don't have a verified recipe for that in my cookbook sources.
   I can only share recipes that come from my trusted published sources."
2. **NEVER fabricate or make up recipes, ingredients, or cooking steps.** Every piece of information
   must come from the retrieved context.
3. **ONLY answer questions about Nigerian indigenous food.** If someone asks about non-Nigerian food,
   politely decline: "I specialize exclusively in Nigerian indigenous soups and food.
   I can't help with that, but ask me about any Nigerian dish!"
4. **Always cite your source** — mention which cookbook the recipe comes from.
5. **Include regional origin** — state which part of Nigeria (and ethnic group) the dish is associated
   with (e.g., "This is a Yoruba dish from Southwest Nigeria" or "This is an Efik delicacy from
   Cross River State").
6. **Use local language names** — include the Yoruba, Igbo, Hausa, Efik, or other local names for
   ingredients and dishes alongside their English names. For example: "iru (locust beans)",
   "ogiri (fermented sesame seeds)", "ugu (fluted pumpkin leaves)".
7. **Provide clear cooking directions** — when giving a recipe, include:
   - List of ingredients with local names
   - Step-by-step cooking instructions
   - Tips and variations if mentioned in the source
   - Serving suggestions

REGIONAL KNOWLEDGE:
- Yoruba (Southwest): Efo Riro, Gbegiri, Ewedu, Ila Asepo, Obe Ata, Amala
- Igbo (Southeast): Ofe Onugbu (Bitterleaf), Oha, Ofe Nsala (White Soup), Egusi, Ogbono, Okazi
- Hausa/Fulani (North): Miyan Kuka, Miyan Taushe, Miyan Yakuwa, Tuwo Shinkafa
- Efik/Ibibio (Cross River): Afang, Edikang Ikong, Ekpang Nkukwo, Afia Efere, Atama
- Edo/Benin: Banga (Palm Fruit Soup), Black Soup (Omoebe), Owo Soup
- Ijaw/Niger Delta: Kekefiyai, Banga, Okoho
- Urhobo/Isoko: Banga (Ofe Akwu variant), Owo, Udu Soup

FORMAT YOUR RESPONSES CLEARLY:
- Use headers and bullet points for readability
- Separate ingredients from instructions
- Bold important terms and local names

Retrieved Context:
{context}

Chat History:
{chat_history}

User Question: {question}

Remember: If the context above does NOT contain relevant recipe information for the user's question,
you MUST say so honestly. Never guess or make up a recipe."""

CONDENSE_PROMPT = """Given the following conversation and a follow-up question, rephrase the
follow-up question to be a standalone question about Nigerian food or recipes.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone Question:"""


def get_api_key():
    """Get OpenAI API key from environment or Streamlit secrets."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # Fall back to environment variable / .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    return None


def init_chain(api_key: str):
    """Initialize the RAG chain with ChromaDB retriever."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=api_key,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )

    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_PROMPT,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
        verbose=False,
    )

    return chain


def format_sources(source_docs):
    """Format source documents for display."""
    if not source_docs:
        return ""

    seen = set()
    sources = []
    for doc in source_docs:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source_name}-p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"- *{source_name}*, page {page}")

    if sources:
        return "\n\n---\n**Sources:**\n" + "\n".join(sources)
    return ""


def main():
    st.set_page_config(
        page_title="Naija Recipes Agent",
        page_icon="🍲",
        layout="centered",
    )

    # Custom styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #008751;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍲 Naija Recipes Agent</h1>
        <p>Your trusted guide to authentic Nigerian indigenous soups and food</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "This agent provides **authentic Nigerian recipes** from trusted, "
            "published cookbook sources only. It will never fabricate a recipe."
        )

        st.markdown("### What can I ask?")
        st.markdown(
            "- How to make a specific Nigerian soup\n"
            "- Ingredients for a dish\n"
            "- Which region a dish comes from\n"
            "- Nigerian soups from a specific region\n"
            "- Local names for ingredients"
        )

        st.markdown("### Try these:")
        st.markdown(
            '- "How do I make Egusi soup?"\n'
            '- "What soups are from Edo State?"\n'
            '- "Give me the recipe for Efo Riro"\n'
            '- "How to prepare Afang soup"\n'
            '- "What is Ofe Nsala?"'
        )

        st.markdown("---")
        st.markdown(
            "**Sources:** All recipes come from verified, published Nigerian cookbooks. "
            "No recipes are fabricated."
        )

    # Check API key
    api_key = get_api_key()
    if not api_key:
        st.error(
            "OpenAI API key not found. Please set it in your `.env` file "
            "or Streamlit Cloud secrets."
        )
        st.stop()

    # Check ChromaDB exists
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        st.error(
            "Recipe knowledge base not found. Please run `python ingest.py` first "
            "to load the cookbook sources."
        )
        st.stop()

    # Initialize chain
    if "chain" not in st.session_state:
        with st.spinner("Loading recipe knowledge base..."):
            st.session_state.chain = init_chain(api_key)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I'm the **Naija Recipes Agent** 🍲\n\n"
                    "I can help you with authentic Nigerian indigenous soup and food recipes. "
                    "All my recipes come from trusted, published Nigerian cookbooks.\n\n"
                    "Ask me about any Nigerian soup or dish — like **Egusi**, **Efo Riro**, "
                    "**Afang**, **Ofe Nsala**, **Banga**, or **Miyan Kuka**!\n\n"
                    "What would you like to cook today?"
                ),
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about any Nigerian dish..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching my cookbooks..."):
                try:
                    result = st.session_state.chain.invoke({"question": prompt})
                    answer = result.get("answer", "I couldn't find an answer.")
                    source_docs = result.get("source_documents", [])

                    # Add source citations
                    full_response = answer + format_sources(source_docs)

                    st.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
