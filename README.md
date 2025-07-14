# Conversational RAG

Conversational RAG is a Python-based chatbot application that leverages Retrieval-Augmented Generation (RAG) techniques to provide intelligent, context-aware responses. The project utilizes FAISS for efficient vector search and Streamlit for an interactive user interface.

## Features
- Conversational chatbot with context retention
- Retrieval-Augmented Generation (RAG) for enhanced answers
- FAISS-based vector database for fast similarity search
- Streamlit-powered web interface
- Modular and extensible codebase

## Directory Structure
```
faiss-db/
  faiss_index_adb/
    index.faiss
    index.pkl
log_chatbot_streamlit.py
readme.md
```
- `faiss-db/`: Contains the FAISS vector database and index files
- `log_chatbot_streamlit.py`: Main Streamlit app for the chatbot
- `readme.md`: Project documentation

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Conversational-RAG
   ```
2. **Install dependencies**
   Ensure you have Python 3.8+ installed. Install required packages:
   ```bash
   pip install streamlit faiss-cpu
   ```
   (Add any other dependencies your project uses)

3. **Prepare the FAISS index**
   - Place your FAISS index files in the `faiss-db/faiss_index_adb/` directory.

## Usage
Run the Streamlit app:
```bash
streamlit run log_chatbot_streamlit.py
```
Open the provided local URL in your browser to interact with the chatbot.

## Requirements
- Python 3.8+
- streamlit
- faiss-cpu
- (Add any other dependencies here)

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
Feel free to customize this README with more details about your project, usage examples, or contribution guidelines as needed.
