from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv
load_dotenv() 

# ----------------------
# FastAPI Setup
# ----------------------
app = FastAPI(title="Virtual Yash")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Request/Response Models
# ----------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# ----------------------
# LangChain RAG Setup
# ----------------------
# Global variables for RAG components
vectorstore = None
qa_chain = None

def initialize_rag_from_pdf(pdf_path: str):
    """Initialize RAG system from a PDF file"""
    global vectorstore, qa_chain, llm_fallback
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(pages)
        
        # Create embeddings and FAISS store
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Build RAG pipeline
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatMistralAI(temperature=0.4, model="mistral-small-latest"),
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff"
        )
        
        # Fallback LLM (direct generation without retrieval)
        llm_fallback = ChatMistralAI(temperature=0.4, model="mistral-small-latest")

        print(f"RAG system initialized successfully from {pdf_path}")
        print(f"Loaded {len(pages)} pages, created {len(docs)} chunks")
        
    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")
        raise

def query_rag_or_llm(question: str):
    """
    Query RAG system first. If no relevant documents found, fallback to direct LLM generation.
    Returns a dictionary with 'answer' and optionally 'source_documents'.
    """
    global qa_chain, vectorstore, llm_fallback
    
    # Search relevant docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    
    if relevant_docs and len(relevant_docs) > 0:
        # Use RAG pipeline
        result = qa_chain.run(question)
        return {"answer": result, "source_documents": relevant_docs}
    else:
        # Fallback to direct LLM generation
        answer = llm_fallback.invoke(question)
        return {"answer": answer, "source_documents": []}


# ----------------------
# API Endpoints
# ----------------------

@app.on_event("startup")
async def startup_event():
    """Initialize RAG on startup"""
    # Check if PDF exists, otherwise use sample text
    pdf_path = r"C:\Users\hp\Downloads\folder\folder\about me.pdf"  # Change this to your PDF path
    
    if os.path.exists(pdf_path):
        initialize_rag_from_pdf(pdf_path)


# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Render your HTML at root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("inedx2.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Run query through RAG pipeline
        result = qa_chain.invoke({"query": request.question})
        answer = result["result"]
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_initialized": qa_chain is not None,
        "vectorstore_ready": vectorstore is not None
    }

# ----------------------
# Run Server
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)