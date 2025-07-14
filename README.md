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
    put your .faiss and .pickle files here.txt
make_vector_db.ipynb
app.py
requirements.txt
run.bat
README.md
```
- `faiss-db/`: Contains the FAISS vector database and index files
- `faiss_index_adb/`: Place your FAISS index files here (see below)
- `make_vector_db.ipynb`: Notebook to create your FAISS index from text data
- `app.py`: Main Streamlit app for the chatbot
- `requirements.txt`: Project dependencies
- `run.bat`: Windows batch file to launch the app
- `README.md`: Project documentation

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Conversational-RAG
   ```
2. **Install dependencies**
   Ensure you have Python 3.8+ installed. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Enter your API key**
   When running the app, you will be prompted to enter your Google Gemini API key in the sidebar. If you do not enter it, the app will use the default placeholder. For best results, obtain your own API key and enter it when prompted.

4. **Prepare the FAISS index**
   - Use `make_vector_db.ipynb` to create your FAISS index from your log or text data. This will generate the necessary `.faiss` and `.pkl` files.
   - Place the generated files in the `faiss-db/faiss_index_adb/` directory. (A placeholder file is present by default.)

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
Or on Windows, you can use:
```bash
run.bat
```
Open the provided local URL in your browser to interact with the chatbot.

## Requirements
- Python 3.8+
- streamlit
- langchain
- langchain-community
- langchain-google-genai
- faiss-cpu
- sentence-transformers
- google-generativeai
- torch

(See `requirements.txt` for the full list.)

---
Feel free to customize this README with more details about your project, usage examples, or contribution guidelines as needed.
