import os
import time
import streamlit as st
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

class ContextAwareMessage(AIMessage):
    """Custom message class that stores both response and context"""
    def __init__(self, content: str, context: str = "", **kwargs):
        super().__init__(content=content, **kwargs)
        self.context = context

class LogChatbot:
    def __init__(self, api_key=None, persist_directory=None):
        os.environ["GOOGLE_API_KEY"] = api_key or os.getenv("GOOGLE_API_KEY", "enter_your_api_key") #<--- Enter your gemini API key
        self.persist_directory = persist_directory or r"faiss-db\faiss_index_adb"
        self._init_components()

    def _init_components(self):
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"FAISS index directory not found: {self.persist_directory}")

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = FAISS.load_local(self.persist_directory, self.embedding_model, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, convert_system_message_to_human=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Updated: prompt rewriting is now independent of chat history
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            """Optimize user query for effective RAG retrieval.

            User Query:  {question} 
            CRITICAL: Make sure to give just the modified query on the output
            """
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """You are an ADB log analysis expert. Use the provided context and chat history to generate a technically accurate and concise answer.
            Format your response using markdown for better readability:
            - Use **bold** for important terms
            - Use `code blocks` for commands, file paths, or technical terms
            - Use bullet points or numbered lists when appropriate
            - Use code fences (```) for multi-line code or log excerpts

                Chat History:
                {chat_history}

                Current Context from Vector Database:
                {context}

                Question:
                {question}

                Answer:"""
        )

        self._build_chain()

    def _build_chain(self):
        self.rag_chain = (
            {"question": RunnablePassthrough()}
            | RunnableLambda(lambda inputs: {
                "original_question": inputs["question"],
                "rewritten_question": self._rewrite_question(inputs["question"]),
            })
            | {
                "context": RunnableLambda(lambda x: self._format_docs(self.retriever.get_relevant_documents(x["rewritten_question"]))),
                "question": RunnableLambda(lambda x: x["original_question"]),
                "chat_history": RunnableLambda(lambda _: self._format_chat_history())
            }
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def _rewrite_question(self, question: str) -> str:
        rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
        return rewrite_chain.invoke({"question": question}).strip()

    def _format_chat_history(self) -> str:
        """Enhanced chat history formatting that includes context from previous interactions"""
        messages = self.memory.chat_memory.messages[-3:]  # Last 3 messages only
        formatted_history = []
        
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, ContextAwareMessage):
                formatted_history.append(f"AI: {message.content}")
                if hasattr(message, 'context') and message.context:
                    formatted_history.append(f"[Previous Context: {message.context}]")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"AI: {message.content}")
        
        return "\n".join(formatted_history)

    def _format_docs(self, docs) -> str:
        """Format documents as a single continuous string without document separators"""
        if not docs:
            return "No relevant documents found."
        
        return " ".join([doc.page_content for doc in docs])
    
    def get_response(self, query: str) -> Tuple[str, Dict[str, Any]]:
        start = time.time()
        try:
            # Rewrite the query explicitly first
            rewritten = self._rewrite_question(query)

            # Get the context that will be used for this query
            relevant_docs = self.retriever.get_relevant_documents(rewritten)
            current_context = self._format_docs(relevant_docs)

            # Now feed it into the chain
            response = self.rag_chain.invoke(query)

            # Store the user message
            self.memory.chat_memory.add_user_message(query)
            
            # Store the AI response WITH the context it was based on
            context_aware_message = ContextAwareMessage(
                content=response,
                context=current_context
            )
            self.memory.chat_memory.messages.append(context_aware_message)

            return response, {
                "query": query,
                "rewritten_query": rewritten,
                "context_used": current_context[:200] + "..." if len(current_context) > 200 else current_context,
                "time": f"{time.time()-start:.2f}s",
                "status": "success"
            }
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e), "status": "failed"}

    def clear_memory(self):
        self.memory.clear()

    def stats(self):
        messages = self.memory.chat_memory.messages
        context_aware_count = sum(1 for m in messages if isinstance(m, ContextAwareMessage))
        return {
            "total": len(messages),
            "user": sum(1 for m in messages if isinstance(m, HumanMessage)),
            "ai": sum(1 for m in messages if isinstance(m, (AIMessage, ContextAwareMessage))),
            "context_aware": context_aware_count
        }

    def health_check(self):
        return {
            "vectorstore": bool(self.vectorstore),
            "llm": bool(self.llm),
            "retriever": bool(self.retriever),
            "memory": bool(self.memory),
            "overall": all([self.vectorstore, self.llm, self.retriever, self.memory])
        }

    def get_last_context(self) -> str:
        """Helper method to get the context from the last AI response"""
        messages = self.memory.chat_memory.messages
        for message in reversed(messages):
            if isinstance(message, ContextAwareMessage) and hasattr(message, 'context'):
                return message.context
        return "No previous context found."

def display_bot_message(content, show_rewritten_query=False, rewritten_query="", show_debug=False, debug_info=None):
    """Helper function to properly display bot messages within the styled container"""
    # Create the complete HTML structure with the bot response content
    bot_html = f"""
    <div class="bot-message-container">
        <div class="bot-message-header">🤖 Bot:</div>
        <div class="bot-content">
            {content}
        </div>
    </div>
    """
    
    # Display the bot message container
    st.markdown(bot_html, unsafe_allow_html=True)
    
    # Show rewritten query if enabled
    if show_rewritten_query and rewritten_query:
        st.markdown(f'<div class="rewritten-query"><strong>Rewritten Query:</strong> {rewritten_query}</div>', unsafe_allow_html=True)
    
    # Show debug info if enabled
    if show_debug and debug_info:
        with st.expander("🔍 Debug Information"):
            st.json(debug_info)

# Streamlit App
def main():
    # Page configuration with light theme
    st.set_page_config(
        page_title="ADB Log Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        color: #212529;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    
    .bot-message-container {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
    }
    
    .bot-message-header {
        color: #2e7d32;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .bot-content {
        color: #2e7d32;
    }
    
    .rewritten-query {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #ff9800;
        color: #e65100;
        font-style: italic;
        font-size: 0.9em;
    }
    
    .debug-info {
        background-color: #fafafa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
        color: #424242;
        font-size: 0.85em;
    }
    
    .sidebar .stSelectbox > label {
        color: #333333;
    }
    
    /* Ensure code blocks in bot responses are visible */
    .bot-message-container pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        overflow-x: auto;
    }
    
    .bot-message-container code {
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    
    .bot-content h1, .bot-content h2, .bot-content h3, 
    .bot-content h4, .bot-content h5, .bot-content h6 {
        color: #1b5e20;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    .bot-content ul, .bot-content ol {
        margin-left: 20px;
    }
    
    .bot-content strong {
        color: #1b5e20;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and header
    st.title("🤖 ADB Log Chatbot")
    st.markdown("**Smart Query with your ADB logs**")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("Google API Key", type="password", help="Enter your Google API key")
        db_path = st.text_input("FAISS DB Path", value="faiss-db\\faiss_index_adb", help="Path to your FAISS index directory")
        
        st.header("🔧 Options")
        show_debug = st.checkbox("Show Debug Info", value=False)
        show_rewritten_query = st.checkbox("Show Rewritten Query", value=True)
        
        st.header("📊 Actions")
        if st.button("Clear Memory", type="secondary"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_memory()
                st.session_state.messages = []
                st.success("Memory cleared!")
        
        if st.button("Health Check", type="secondary"):
            if 'chatbot' in st.session_state:
                health = st.session_state.chatbot.health_check()
                if health["overall"]:
                    st.success("All systems operational!")
                else:
                    st.error("Some components failed")
                st.json(health)
        
        if st.button("Show Stats", type="secondary"):
            if 'chatbot' in st.session_state:
                stats = st.session_state.chatbot.stats()
                st.json(stats)

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = LogChatbot(api_key=api_key, persist_directory=db_path)
            st.success("Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.stop()

    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Use the helper function to display bot messages properly
            display_bot_message(
                content=message["content"],
                show_rewritten_query=show_rewritten_query,
                rewritten_query=message.get("rewritten_query", ""),
                show_debug=show_debug,
                debug_info=message.get("debug_info", None)
            )

    # Chat input
    if prompt := st.chat_input("Ask about ADB logs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
        
        # Get bot response
        with st.spinner("Thinking..."):
            try:
                response, debug_info = st.session_state.chatbot.get_response(prompt)
                
                # Add assistant response to chat history
                message_data = {
                    "role": "assistant", 
                    "content": response,
                    "debug_info": debug_info
                }
                
                if "rewritten_query" in debug_info:
                    message_data["rewritten_query"] = debug_info["rewritten_query"]
                
                st.session_state.messages.append(message_data)
                
                # Display bot response using the helper function
                display_bot_message(
                    content=response,
                    show_rewritten_query=show_rewritten_query,
                    rewritten_query=debug_info.get("rewritten_query", ""),
                    show_debug=show_debug,
                    debug_info=debug_info
                )
                        
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:** Use the sidebar to configure settings, clear memory, or check system health.")

if __name__ == "__main__":
    main()