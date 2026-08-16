import os
import glob # this helps in storing all pdfs files together in a list or an array
import warnings
from dotenv import load_dotenv # this is used to load the api key which plays an important role in the project i.e communicating with the llm
# i am using fast api web server which will connect my rag system to the frontend 
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel # this helps in ensuring whether the session id of user is generated and whether the user has given the questions as an input 

from langchain_community.document_loaders import PyPDFLoader # this is used to read and open pdf files
from langchain_text_splitters import RecursiveCharacterTextSplitter # converts lon texts into small chunks or overlapping cards
from langchain_huggingface import HuggingFaceEmbeddings #embedding models chose , which converts text cards into vector numbers
from langchain_chroma import Chroma # vector database
from langchain_google_genai import ChatGoogleGenerativeAI #answers the user's questions
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda # this is used to provide event listeners
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # our model also has conversational memory
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

app = FastAPI(title="Your Local Search Assistant") # app is the main varibale which is used ahead to run server also while executing the code through terminal


store = {} # dictionary for storing session ids
rag_chain = None
# self-explanatory code below
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def initialize_rag_system():
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("❌ No PDF files found in 'data/' directory.")
        return None

    all_pages = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        all_pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_pages)

    print("⚡ Initializing vector database...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) # k here respresents no of text cards

    print("🤖 Connecting to Google Gemini API...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", # we are using latest model -> gemini-3.7-flash
        temperature=0.2 # gives more logical answers than creative answers
    )
    # reformulation of the question based on past history chat of user 
    # template 1
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    question_rewriter = contextualize_q_prompt | llm | StrOutputParser() # chain formation
    # decomposition of multi-part questions into stand - alone questions
    decomposition_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Break down the user's input into separate single standalone sub-questions if it contains multiple questions or distinct topics. Return each sub-question on a new line. If it is already a single question, return it as is."),
        ("human", "{question}")
    ])
    query_decomposer = decomposition_prompt | llm | StrOutputParser()
    # template 2
    template = """You are a helpful research assistant. 
Answer the user's question accurately using ONLY the provided context. 
If the answer to any part cannot be found in the context, state clearly which parts could not be found.

Context:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    # self - explanatory code 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_with_sources(input_data):
        raw_question = input_data["question"]
        chat_history = input_data.get("chat_history", [])
        
        if chat_history:
            standalone_question = question_rewriter.invoke({
                "chat_history": chat_history,
                "question": raw_question
            })
        else:
            standalone_question = raw_question
            
        sub_queries_raw = query_decomposer.invoke({"question": standalone_question})
        sub_queries = [q.strip() for q in sub_queries_raw.split("\n") if q.strip()]
        
        retrieved_docs = [] # de-duplication(prevents redundancy)
        doc_ids = set()
        
        for query in sub_queries:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in doc_ids:
                    doc_ids.add(doc.page_content)
                    retrieved_docs.append(doc)
        
        formatted_context = format_docs(retrieved_docs)
        
        citations = []
        for doc in retrieved_docs:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown")) # name of the pdf
            page_num = doc.metadata.get("page", 0) + 1 # page no.
            citation_str = f"• {source_file} (Page {page_num})"
            if citation_str not in citations: # prevents similar type of citations to get printed
                citations.append(citation_str)

        qa_chain = qa_prompt | llm | StrOutputParser()
        llm_response = qa_chain.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "question": raw_question
        })

        sources_text = "\n\n📌 **Sources Used:**\n" + "\n".join(citations)
        return llm_response + sources_text

    return RunnableWithMessageHistory(
        runnable=RunnableLambda(rag_with_sources),
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
# self explanatory code 
@app.on_event("startup")
def startup_event():
    global rag_chain
    rag_chain = initialize_rag_system()

# Request model for API
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system failed to initialize.")
    
    try:
        response = rag_chain.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")
# end of the code . Thankyou!!