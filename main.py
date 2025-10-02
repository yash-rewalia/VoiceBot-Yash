import os
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# ----------------------
# Configuration
# ----------------------
load_dotenv()
PDF_PATH = "about me.pdf"
FAISS_INDEX_PATH = "faiss_index"
TEMPLATES_DIR = "templates"
LLM_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"

# ----------------------
# Logging Setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

# ----------------------
# FastAPI Setup
# ----------------------
app = FastAPI(title="RAG + LLM Chat Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ----------------------
# Request/Response Models
# ----------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# ----------------------
# RAG + LLM Service
# ----------------------
class RAGService:
    def __init__(self, pdf_path: str, faiss_path: str):
        self.pdf_path = pdf_path
        self.faiss_path = faiss_path
        self.embeddings = MistralAIEmbeddings(model=EMBED_MODEL)
        self.llm = ChatMistralAI(temperature=0.4, model=LLM_MODEL)

        # Chains
        self.vectorstore = None
        self.rag_chain = None
        self.general_llm_chain = None

        self._initialize()

    def _initialize(self):
        # ---------------------
        # RAG Chain Initialization
        # ---------------------
        if os.path.exists(self.faiss_path):
            logging.info("FAISS index found. Loading vectorstore from disk.")
            try:
                self.vectorstore = FAISS.load_local(
                    self.faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logging.error(f"Failed to load FAISS index: {e}")
                self.vectorstore = None
        elif os.path.exists(self.pdf_path):
            logging.info(f"FAISS index not found. Creating new vectorstore from PDF: {self.pdf_path}")
            try:
                loader = PyPDFLoader(self.pdf_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(pages)
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
                self.vectorstore.save_local(self.faiss_path)
                logging.info(f"Vectorstore created and saved to {self.faiss_path}")
            except Exception as e:
                logging.error(f"Failed to create vectorstore from PDF: {e}")
                self.vectorstore = None
        else:
            logging.warning(f"PDF not found at {self.pdf_path}. RAG chain not initialized.")
            self.vectorstore = None

        if self.vectorstore:
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

                # Use a proper prompt for RAG
                rag_prompt = PromptTemplate(
                    template="""
                    You are a knowledgeable assistant. Begin your response with a warm greeting. Use the provided context to answer the user’s question, but explain it in a natural, conversational way rather than just restating the context. Make sure the reply feels human-like and approachable. At the end of your response, include an open-ended question to keep the conversation flowing
                    For general greeting reply in short only.

                    Input:
                    Context: {context}
                    Question: {question}

                    Output:
                    """,
                    input_variables=["context", "question"]
                )

                self.rag_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever,
                    return_source_documents=False,  # important: return string, not dict
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": rag_prompt}
                )
                logging.info("RAG chain initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize RAG chain: {e}")
                self.rag_chain = None

        # ---------------------
        # General LLM Chain Initialization
        # ---------------------
        try:
            prompt = PromptTemplate(
                template="""You are a knowledgeable assistant.

                    - If the user sends a general greeting (e.g., “hi”, “hello”, “hey”), reply briefly with a warm, friendly greeting only. Do not pull from context in this case.
                    - If the user asks a question:
                        - Begin with a warm greeting.
                        - Use the provided context to answer naturally and conversationally (do not just restate context).
                        - Ensure the tone feels approachable and human-like.
                        - End with an open-ended question to keep the conversation flowing.

                    Input:
                    Question: {question}

                    Output:
                    """,
                input_variables=["question"]
            )
            self.general_llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            logging.info("General LLM chain initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize general LLM chain: {e}")
            self.general_llm_chain = None

    # ---------------------
    # Answering Logic
    # ---------------------
    def answer(self, question: str) -> str:
        if not question.strip():
            return "Question is empty."

        # Detect greetings
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        normalized = question.strip().lower()
        if any(normalized == g or normalized.startswith(g + " ") for g in greetings):
            logging.info("Detected greeting, using general LLM chain only.")
            if self.general_llm_chain:
                try:
                    llm_answer = self.general_llm_chain.run(question)
                    logging.info(f"LLM greeting answer: {llm_answer}")
                    return llm_answer
                except Exception as e:
                    logging.error(f"General LLM chain failed: {e}")
                    return "Sorry, I could not answer that."
            else:
                return "LLM chain not initialized."

        # 1️⃣ Try RAG first
        rag_answer = None
        if self.rag_chain:
            try:
                rag_answer = self.rag_chain.run(question)
                rag_answer = rag_answer.strip() if rag_answer else None
                logging.info(f"RAG answer: {rag_answer}")
            except Exception as e:
                logging.error(f"RAG chain failed: {e}")
                rag_answer = None

        # 2️⃣ Fallback only if RAG failed
        if not rag_answer:
            logging.info("Fallback to general LLM chain.")
            if self.general_llm_chain:
                try:
                    llm_answer = self.general_llm_chain.run(question)
                    logging.info(f"LLM fallback answer: {llm_answer}")
                    return llm_answer
                except Exception as e:
                    logging.error(f"General LLM chain failed: {e}")
                    return "Sorry, I could not answer that."
            else:
                return "LLM chain not initialized."

        return rag_answer

# ----------------------
# Singleton Dependency
# ----------------------
rag_service = RAGService(PDF_PATH, FAISS_INDEX_PATH)

def get_rag_service():
    return rag_service

# ----------------------
# API Endpoints
# ----------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RAGService = Depends(get_rag_service)):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = rag.answer(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(rag: RAGService = Depends(get_rag_service)):
    return {
        "status": "healthy",
        "rag_initialized": rag.rag_chain is not None,
        "vectorstore_ready": rag.vectorstore is not None,
        "general_llm_ready": rag.general_llm_chain is not None
    }

# ----------------------
# Run Server
# ----------------------
# if __name__ == "__main__":
#     logging.info("Starting server on 0.0.0.0:8800")
#     uvicorn.run(app, host="0.0.0.0", port=8800)
