import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DB_PATH = "chroma_db"
st.set_page_config(page_title="Tech Manual Assistant", layout="wide")
st.title("🔧 Offline Technical Assistant")
st.caption("⚡ Powered by Local Llama 3.2 | 🧠 Context-Aware Memory")

# ---------------------------------------------------------
# CUSTOM RAG WITH MEMORY
# ---------------------------------------------------------
class ConversationalRAG:
    """Custom RAG system with conversation memory"""
    
    def __init__(self, vectorstore, llm, max_history=6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_history = max_history  # Keep last N exchanges
        self.conversation_history = []
    
    def _build_context_from_docs(self, docs):
        """Extract text from retrieved documents"""
        return "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
    
    def _build_conversation_context(self):
        """Format recent conversation history"""
        if not self.conversation_history:
            return "No previous conversation."
        
        # Get last max_history messages
        recent = self.conversation_history[-self.max_history:]
        formatted = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def _needs_reformulation(self, question):
        """
        Use LLM to intelligently detect if question references previous conversation.
        More accurate than simple keyword matching.
        """
        if not self.conversation_history:
            return False
        
        # Get last 2 exchanges for context
        recent = self.conversation_history[-4:] if len(self.conversation_history) >= 4 else self.conversation_history
        
        if not recent:
            return False
        
        history_snippet = "\n".join([
            f"{msg['role'].title()}: {msg['content']}" 
            for msg in recent
        ])
        
        detection_prompt = f"""Recent conversation:
{history_snippet}

New question: "{question}"

Does this new question reference or depend on information from the conversation above? Consider:
- Does it use pronouns (it, that, they, etc.) referring to previous topics?
- Does it ask for clarification or follow-up on a previous answer?
- Is it a continuation of the previous discussion?
- Would someone reading ONLY this question understand what's being asked?

Answer with ONLY one word: YES or NO"""
        
        try:
            response = self.llm.invoke(detection_prompt)
            answer = response.content.strip().upper()
            return "YES" in answer
        except:
            # Fallback to keyword-based detection if LLM fails
            reference_words = ["it", "that", "this", "they", "them", "those", "what about", "how about", "and the", "also"]
            return any(word in question.lower() for word in reference_words)
    
    def _reformulate_question(self, question):
        """
        If the question references previous context (e.g., "what about that?"),
        reformulate it into a standalone question using conversation history.
        """
        if not self.conversation_history:
            return question
        
        # Use intelligent detection
        if not self._needs_reformulation(question):
            return question
        
        # Use LLM to reformulate with richer context
        history_context = self._build_conversation_context()
        reformulation_prompt = f"""You are helping reformulate a follow-up question into a standalone question.

Conversation history:
{history_context}

Current question: "{question}"

Task: Rewrite this question so it can be understood WITHOUT the conversation history. Include all necessary context from the conversation in the reformulated question.

Examples:
- "What about that?" → "What is the torque specification for the secondary bolt?"
- "How do I install it?" → "How do I install the fuel pump assembly?"
- "Tell me more" → "Tell me more about the hydraulic system maintenance procedures"

Reformulated standalone question:"""
        
        try:
            response = self.llm.invoke(reformulation_prompt)
            reformulated = response.content.strip()
            # Remove quotes if LLM added them
            reformulated = reformulated.strip('"').strip("'")
            return reformulated
        except:
            return question  # Fallback to original if reformulation fails
    
    def query(self, question):
        """
        Main query method:
        1. Reformulate question if it references conversation history
        2. Retrieve relevant documents
        3. Generate answer with full context
        4. Store in conversation memory
        """
        # Step 1: Reformulate if needed
        standalone_question = self._reformulate_question(question)
        
        # Step 2: Retrieve relevant documents
        docs = self.vectorstore.similarity_search(standalone_question, k=3)
        doc_context = self._build_context_from_docs(docs)
        
        # Step 3: Build final prompt with everything
        conversation_context = self._build_conversation_context()
        
        final_prompt = f"""You are an expert technical assistant for GE Aerospace manuals.

Previous Conversation:
{conversation_context}

Relevant Manual Excerpts:
{doc_context}

Current Question: {question}

Instructions:
- Use the manual excerpts above to answer the question
- Reference previous conversation if relevant
- If you don't know, say so clearly
- Keep answer concise (max 4 sentences)
- Be specific and technical when appropriate

Answer:"""
        
        # Step 4: Generate answer
        response = self.llm.invoke(final_prompt)
        answer = response.content.strip()
        
        # Step 5: Store in memory
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "sources": docs,
            "reformulated_question": standalone_question if standalone_question != question else None
        }
    
    def clear_history(self):
        """Reset conversation memory"""
        self.conversation_history = []
    
    def get_history(self):
        """Get full conversation history"""
        return self.conversation_history

# ---------------------------------------------------------
# INITIALIZE COMPONENTS
# ---------------------------------------------------------
@st.cache_resource
def load_components():
    """Load vectorstore and LLM (cached for performance)"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = ChatOllama(model="llama3.2:1b", temperature=0.3)
    return db, llm

db, llm = load_components()

# Initialize RAG system in session state
if "rag" not in st.session_state:
    st.session_state.rag = ConversationalRAG(db, llm, max_history=6)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# SIDEBAR - MEMORY CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header("💬 Conversation Memory")
    
    total_exchanges = len(st.session_state.rag.conversation_history) // 2
    st.metric("Exchanges Stored", total_exchanges)
    
    if st.button("🗑️ Clear Memory", use_container_width=True):
        st.session_state.rag.clear_history()
        st.session_state.messages = []
        st.rerun()
    
    with st.expander("📚 How Memory Works"):
        st.markdown("""
        **This assistant remembers:**
        - Your last 6 message exchanges
        - Automatically reformulates follow-up questions
        - Uses context from previous answers
        
        **Example:**
        - You: "What's the torque spec?"
        - Bot: "50 Nm for the main bolt"
        - You: "What about the secondary bolt?" ← Uses context!
        """)

# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask about the technical manual..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching manual and analyzing context..."):
            result = st.session_state.rag.query(prompt)
            answer = result["answer"]
            
            st.markdown(answer)
            
            # Show if question was reformulated (helps user understand)
            if result["reformulated_question"]:
                with st.expander("🔄 Question Reformulation"):
                    st.info(f"Interpreted as: *{result['reformulated_question']}*")
            
            # Show sources
            with st.expander("📄 Sources"):
                for i, doc in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc.page_content[:200] + "...")
                    st.divider()
    
    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("💡 Tip: Ask follow-up questions! The assistant remembers your conversation.")