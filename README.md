# Hybrid-Search-RAG-System

Combining a semantic search + keyword search approach to easily retrieve information from your local documents.

Ensure your project directory matches the structure below:

```text
SMART-AI-ASSISTANT/
├── app/
│   ├── static/
│   │   └── index.html
│   └── main.py
├── data/
│   └── (Put your PDF files here)
├── .env
├── .gitignore
└── requirements.txt
```
#### Local Setup & Installation
 Clone the Repository
 
git clone <REPOSITORY_URL>
(Copy the URL under the HTTPS or SSH tab on GitHub depending on your SSH key configuration.)

#### Create virtual environment
python -m venv venv  # Use python3 depending on your installation

Activate the virtual environment:

#### Windows:

venv\Scripts\activate

#### Mac / Linux:

source venv/bin/activate

#### Install all required Python packages:

pip install -r requirements.txt

#### Configure API Keys
 
Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY=your_google_api_key_here

Note: Obtain a free API key from Google AI Studio. This project uses the latest gemini-3.7-flash model.

#### Run the Application
Start the FastAPI server using Uvicorn:

uvicorn app.main:app --reload

Once running, your terminal will output a local server URL:

http://127.0.0.1:8000

Click or open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser to access the app.

#### Technical Details & Architecture

Keyword Search: Uses the BM25 algorithm to index individual words within your documents.

Semantic + Keyword Hybrid Retrieval: Combines results using Reciprocal Rank Fusion (RRF) via an Ensemble Retriever to find the most accurate answer for the query.

Vector Database: Uses Chroma DB for semantic vector storage. chroma_db/ is added to .gitignore so your local database remains private and fresh databases can be created locally.

Conversational Memory: Uses an SQLite database stored on your hard disk to store user chat history across interactions.

Document Capacity & Query Limits: Works best when uploading 10 to 20 documents inside the data/ folder. You can ask up to 3 questions at a time to the agent across different PDFs.

#### Troubleshooting
ModuleNotFoundError: No module named 'fastapi'

If you encounter a FastAPI module error while running the server, run the following command in your terminal:

pip install "fastapi[standard]" uvicorn

Important: Ensure your virtual environment (venv) is activated before running this command. This issue frequently occurs if you move your project folder to another location (e.g., a pen drive).

#### Project Purpose
Rather than creating a hosted public link, this project is shared as a cloneable repository so developers can inspect what happens under the hood, learn classic RAG system mechanics, and modify or build upon the code.
