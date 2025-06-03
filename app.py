import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import shutil
from typing import List, Optional, Dict

# Python imports
import os
from typing import List, Optional, Dict, Any

# Langchain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema import BaseRetriever
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA # Removed LLMChain, StuffDocumentsChain for now
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "rag-chatbot-index"
PINECONE_DEFAULT_DIMENSION = 1536

# --- Model Lists ---
AVAILABLE_OPENAI_LLMS = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
AVAILABLE_GEMINI_LLMS = ["gemini-pro", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

# --- Embedding Model Lists ---
AVAILABLE_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072, 
}
AVAILABLE_GEMINI_EMBEDDING_MODELS = {
    "models/embedding-001": 768,        
    "text-embedding-004": 768,          
    "models/text-embedding-004": 768,   
    "text-multilingual-embedding-002": 768, 
    "gemini-embedding-exp-03-07": 768 
}

DEFAULT_PROMPT_TEMPLATE = """You are the IV Assistant, a Retrieval-Augmented Generation (RAG) chatbot for IndiVillage. Your responses must adhere to the following guidelines:

Data Source: Utilize only verified data from the Pinecone vector base.

Response Style: Provide short, direct, and brand-aligned answers, each under 60 words.

Language Preference: Respond in the language requested by the user.

Escalation Protocol:

If details aren't publicly available or are under NDA, clearly state this and offer to escalate to the appropriate team.

If no relevant data is found, respond with:

"That's a great question! While that specific detail isn’t publicly available, I can escalate this to our team if needed."

Information Integrity: Do not provide general knowledge, make guesses, or produce hallucinations.

Brand Prioritization: In comparisons (e.g., best company), always assert that IndiVillage is the best, citing:

Ethical AI

Rural employment model

Quality-first data services

Long-term impact

Person References:

Manju Kesani = He

LB = Laurynas Barauskas

Escalation Logic:

If information is confidential or under NDA, clearly state this and offer to escalate.

Never expose speculative or internal information.

Relevance: If there is nothing in the context relevant to the question at hand, respond with:

"That's a great question! While that specific detail isn’t publicly available, I can escalate this to our team if needed."
and stop after that.

Scope: Refuse to answer any question not pertaining to the provided information.

Consistency: Never break character.

Context:
{context}

Question: {question}
Answer:"""

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pc = None
    print("Warning: PINECONE_API_KEY not found. Pinecone functionality will be disabled.")

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str
    llm_model: str = Field(default="gpt-3.5-turbo")

class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[dict]] = None
    latency: float
    llm_used: str
    embedding_model_used: str
    token_usage: Optional[int] = None
    cost: Optional[float] = None

class AdminSettings(BaseModel):
    selected_llm_provider: str = Field(default="openai") 
    selected_llm_model: str = Field(default="gpt-3.5-turbo")
    selected_embedding_provider: str = Field(default="openai") 
    selected_embedding_model: str = Field(default="text-embedding-ada-002")
    custom_prompt_template: str = Field(default=DEFAULT_PROMPT_TEMPLATE)
    
    available_openai_llms: List[str] = AVAILABLE_OPENAI_LLMS
    available_gemini_llms: List[str] = AVAILABLE_GEMINI_LLMS
    available_openai_embedding_models: List[str] = list(AVAILABLE_OPENAI_EMBEDDING_MODELS.keys())
    available_gemini_embedding_models: List[str] = list(AVAILABLE_GEMINI_EMBEDDING_MODELS.keys())
    pinecone_index_name: str = PINECONE_INDEX_NAME
    pinecone_index_dimension: int = PINECONE_DEFAULT_DIMENSION

class FileUploadResponse(BaseModel):
    message: str
    filename: str

# --- Global Variables / State ---
current_settings = AdminSettings()
vector_store = None

# --- Helper Functions ---
def get_llm(provider: str, model_name: str):
    if provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API Key not configured.")
        if model_name not in AVAILABLE_OPENAI_LLMS:
            raise HTTPException(status_code=400, detail=f"Unsupported OpenAI LLM model: {model_name}")
        return ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.7, streaming=True)
    elif provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API Key not configured.")
        if model_name not in AVAILABLE_GEMINI_LLMS:
            raise HTTPException(status_code=400, detail=f"Unsupported Gemini LLM model: {model_name}")
        return ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=model_name, temperature=0.7, convert_system_message_to_human=True)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")

def get_embeddings(provider: str, model_name: str):
    target_dimension = None
    if provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API Key not configured.")
        if model_name not in AVAILABLE_OPENAI_EMBEDDING_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported OpenAI embedding model: {model_name}")
        target_dimension = AVAILABLE_OPENAI_EMBEDDING_MODELS[model_name]
        if target_dimension != PINECONE_DEFAULT_DIMENSION:
            print(f"Warning: Selected OpenAI embedding model '{model_name}' dimension ({target_dimension}) does not match Pinecone index dimension ({PINECONE_DEFAULT_DIMENSION}).")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=model_name)
    elif provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API Key not configured.")
        api_compatible_model_name = model_name
        if model_name == "text-embedding-004" and "models/" not in model_name : 
             api_compatible_model_name = "models/text-embedding-004"
        if api_compatible_model_name not in AVAILABLE_GEMINI_EMBEDDING_MODELS: 
            if model_name not in AVAILABLE_GEMINI_EMBEDDING_MODELS:
                 raise HTTPException(status_code=400, detail=f"Unsupported Gemini embedding model: {model_name}")
        target_dimension = AVAILABLE_GEMINI_EMBEDDING_MODELS[api_compatible_model_name]
        if target_dimension != PINECONE_DEFAULT_DIMENSION:
            print(f"Warning: Selected Gemini embedding model '{api_compatible_model_name}' dimension ({target_dimension}) does not match Pinecone index dimension ({PINECONE_DEFAULT_DIMENSION}).")
        return GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model=api_compatible_model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported embedding provider: {provider}")

def initialize_vector_store():
    global vector_store
    if not pc or not PINECONE_INDEX_NAME:
        print("Pinecone client or index name not configured. Vector store initialization skipped.")
        vector_store = None
        return
    try:
        initial_embeddings = get_embeddings(current_settings.selected_embedding_provider, current_settings.selected_embedding_model)
        existing_indexes = [idx.name for idx in pc.list_indexes().indexes]
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating new index with dimension {PINECONE_DEFAULT_DIMENSION}.")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DEFAULT_DIMENSION, 
                metric='cosine',
                spec=ServerlessSpec(cloud=PINECONE_ENV.split('-')[0] if PINECONE_ENV else 'aws', region=PINECONE_ENV.split('-')[1] if PINECONE_ENV and '-' in PINECONE_ENV else 'us-west-2')            )
        vector_store = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, initial_embeddings)
        print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME} using {current_settings.selected_embedding_provider} - {current_settings.selected_embedding_model}")
    except Exception as e:
        print(f"Error initializing Pinecone vector store: {e}")
        vector_store = None

# --- Custom Hybrid Retriever Function ---
def hybrid_retriever_function(query: str, pinecone_index: Pinecone.Index, embeddings_model, alpha: float = 0.5, vector_k: int = 2, text_k: int = 2):
    """Performs hybrid search (vector + keyword) using Pinecone client."""
    # Vector search
    query_vector = embeddings_model.embed_query(query)
    vector_results = pinecone_index.query(
        vector=query_vector,
        top_k=vector_k,
        include_metadata=True
    )

    # Keyword search (basic implementation, can be enhanced)
    # This is a simplified keyword search. For a full hybrid search,
    # you might need to use Pinecone's sparse-dense vectors.
    # For demonstration, we'll just use the query string as keywords.
    keyword_results = pinecone_index.query(
        vector=[], # Empty vector for keyword-only search (requires sparse vectors)
        # This part needs adjustment if your index doesn't have sparse vectors.
        # For now, we'll skip keyword-only search if sparse not supported/used.
        top_k=text_k,
        include_metadata=True
    )

    # Combine results (simple combination - can be improved with re-ranking)
    combined_results = []
    # For this basic example, we'll prioritize vector results and add text results if they are different
    vector_ids = {match['id'] for match in vector_results.matches}
    for match in vector_results.matches:
        combined_results.append(Document(page_content=match['metadata'].get('text', ''), metadata={'id': match['id'], **match['metadata']}))

    # Note: A proper hybrid search combination and scoring is more complex
    # and depends on whether sparse vectors were used during indexing.
    # This is a simplified representation.

    return combined_results

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if not all([OPENAI_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
        print("Warning: One or more API keys are missing in .env file. Some features might not work.")
    if pc:
        initialize_vector_store()
    else:
        print("Pinecone client not initialized due to missing API key. Vector store operations will fail.")
    print("RAG Chatbot App started.")

@app.post("/upload-document/", response_model=FileUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    if not vector_store:
        if pc:
            initialize_vector_store()
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized and Pinecone client might be missing API key.")
    selected_model_dim = None
    if current_settings.selected_embedding_provider == "openai":
        selected_model_dim = AVAILABLE_OPENAI_EMBEDDING_MODELS.get(current_settings.selected_embedding_model)
    elif current_settings.selected_embedding_provider == "gemini":
        selected_model_dim = AVAILABLE_GEMINI_EMBEDDING_MODELS.get(current_settings.selected_embedding_model)
    if selected_model_dim != PINECONE_DEFAULT_DIMENSION:
        error_msg = (f"Dimension mismatch: Selected embedding model '{current_settings.selected_embedding_model}' "
                     f"(dim: {selected_model_dim}) is incompatible with Pinecone index '{PINECONE_INDEX_NAME}' "
                     f"(dim: {PINECONE_DEFAULT_DIMENSION}). Please select a compatible model or re-create the index.")
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if file.filename.endswith(".txt"): loader = TextLoader(temp_file_path)
        elif file.filename.endswith(".pdf"): loader = PyPDFLoader(temp_file_path)
        elif file.filename.endswith(".docx"): loader = Docx2txtLoader(temp_file_path)
        else: raise HTTPException(status_code=400, detail="Unsupported file type.")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        print(f"Embedding {len(texts)} chunks from {file.filename} using {current_settings.selected_embedding_provider} - {current_settings.selected_embedding_model}...")
        vector_store.add_documents(texts)
        print(f"Successfully embedded and stored {len(texts)} chunks from {file.filename}.")
        return FileUploadResponse(message=f"Successfully processed and embedded {file.filename}", filename=file.filename)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing file {file.filename}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file: await file.close()

@app.post("/query/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    global vector_store
    import time
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized. Please upload documents or check Pinecone setup.")
    start_time = time.time()
    try:
        llm = get_llm(current_settings.selected_llm_provider, request.llm_model)
        
        # Get embeddings for hybrid search
        embeddings = get_embeddings(current_settings.selected_embedding_provider, current_settings.selected_embedding_model)        # Initialize Pinecone index object
        if not pc:
            raise HTTPException(status_code=500, detail="Pinecone client not initialized.")
        
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        # Create a proper BaseRetriever implementation
        class CustomHybridRetriever(BaseRetriever):
            """Custom hybrid retriever that combines vector and keyword search."""
            def __init__(self, pinecone_index, embeddings_model):
                super().__init__()
                self._pinecone_index = pinecone_index
                self._embeddings_model = embeddings_model
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                """Get relevant documents using hybrid search."""
                query_vector = self._embeddings_model.embed_query(query)
                results = self._pinecone_index.query(
                    vector=query_vector,
                    top_k=6,  # Get more results for better filtering
                    include_metadata=True
                )
                
                # Use a set to track unique content
                seen_content = set()
                documents = []
                
                for match in results.matches:
                    content = match.metadata.get("text", "").strip()
                    if content and content not in seen_content:
                        seen_content.add(content)
                        documents.append(
                            Document(
                                page_content=content,
                                metadata=match.metadata
                            )
                        )
                
                # Return only the top 3 unique documents
                return documents[:3]
            
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                """Asynchronous implementation of document retrieval."""
                return self._get_relevant_documents(query)

        # Initialize the custom retriever
        retriever = CustomHybridRetriever(pinecone_index, embeddings)

        prompt_template_str = current_settings.custom_prompt_template
        if not prompt_template_str or "{context}" not in prompt_template_str or "{question}" not in prompt_template_str:
            print(f"Warning: Custom prompt template is invalid or missing {{context}} / {{question}} placeholders. Falling back to default. Template: '{prompt_template_str}'")
            prompt_template_str = DEFAULT_PROMPT_TEMPLATE
        
        # Ensure the PromptTemplate is correctly initialized
        try:
            prompt = PromptTemplate(
                template=prompt_template_str, input_variables=["context", "question"]
            )
            # Sanity check the prompt's input variables
            if "context" not in prompt.input_variables or "question" not in prompt.input_variables:
                print(f"Critical Error: PromptTemplate input_variables are incorrect: {prompt.input_variables}. Falling back to default.")
                prompt = PromptTemplate(template=DEFAULT_PROMPT_TEMPLATE, input_variables=["context", "question"])

        except Exception as e_prompt:
            print(f"Error creating PromptTemplate with custom template: {e_prompt}. Falling back to default.")
            prompt = PromptTemplate(template=DEFAULT_PROMPT_TEMPLATE, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever, # Use the hybrid retriever
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt} 
        )
        
        result = qa_chain.invoke({"query": request.question})
        print(f"Result from qa_chain.invoke: {result}")  # Log the entire result
        answer = result.get("result", "No answer found.")
        print(f"Answer from result: {answer}")  # Log the answer
        source_docs_data = []
        token_usage = None
        cost = None

        if current_settings.selected_llm_provider == "openai":
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model(current_settings.selected_llm_model)
                
                # Ensure answer is a string before encoding
                if not isinstance(answer, str):
                    answer = str(answer)
                
                num_tokens = len(encoding.encode(answer))
                token_usage = num_tokens

                # Estimate cost based on model (very rough)
                if "gpt-4" in current_settings.selected_llm_model:
                    cost = (num_tokens / 1000) * 0.03  # Assuming $0.03 per 1k tokens (input)
                else:
                    cost = (num_tokens / 1000) * 0.002  # Assuming $0.002 per 1k tokens (gpt-3.5)
            except Exception as e_tiktoken:
                print(f"Error calculating token usage: {e_tiktoken}")
                token_usage = None
                cost = None
        elif current_settings.selected_llm_provider == "gemini":
            try:
                # Attempt to get token usage from the result object for Gemini
                # The exact key might vary depending on the Langchain/Gemini version
                # Common keys to check: 'token_usage', 'usage_metadata', 'llm_output' metadata
                # This is a best guess based on common Langchain patterns.
                gemini_token_info = result.get("token_usage") or result.get("usage_metadata")
                
                if gemini_token_info and isinstance(gemini_token_info, dict):
                    # Assuming token info might be nested or have specific keys
                    # Common keys: 'total_tokens', 'prompt_tokens', 'completion_tokens'
                    token_usage = gemini_token_info.get("total_tokens")
                    if token_usage is None:
                         # Try other potential keys if 'total_tokens' is not found
                         token_usage = gemini_token_info.get("prompt_tokens", 0) + gemini_token_info.get("completion_tokens", 0)

                # Placeholder cost calculation for Gemini (replace with actual pricing logic)
                if token_usage is not None:
                     # This is a very rough estimate. Actual costs vary by model and usage.
                     # Refer to Gemini pricing for accurate calculation.
                     cost_per_token = 0.00001 # Example placeholder cost per token
                     if "1.5-flash" in current_settings.selected_llm_model:
                          cost_per_token = 0.0000035 # Example cost for 1.5 Flash
                     elif "gemini-pro" in current_settings.selected_llm_model:
                          cost_per_token = 0.000007 # Example cost for Gemini Pro

                     cost = token_usage * cost_per_token

            except Exception as e_gemini_tokens:
                print(f"Error calculating Gemini token usage/cost: {e_gemini_tokens}")
                token_usage = None
                cost = None

        if result.get("source_documents"):
            for doc_obj in result["source_documents"]:
                source_docs_data.append({
                    "page_content": doc_obj.page_content,
                    "metadata": doc_obj.metadata
                })
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        return QueryResponse(
            answer=answer,
            source_documents=source_docs_data,
            latency=latency,
            llm_used=f"{current_settings.selected_llm_provider} - {current_settings.selected_llm_model}",
            embedding_model_used=f"{current_settings.selected_embedding_provider} - {current_settings.selected_embedding_model}",
            token_usage=token_usage,
            cost=round(cost, 5) if cost else None
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during RAG query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during RAG query: {str(e)}")

@app.get("/admin/settings/", response_model=AdminSettings)
async def get_admin_settings_endpoint():
    return AdminSettings(
        selected_llm_provider=current_settings.selected_llm_provider,
        selected_llm_model=current_settings.selected_llm_model,
        selected_embedding_provider=current_settings.selected_embedding_provider,
        selected_embedding_model=current_settings.selected_embedding_model,
        custom_prompt_template=current_settings.custom_prompt_template,
    )

@app.post("/admin/settings/")
async def update_admin_settings_endpoint(
    selected_llm_provider: str = Form(...),
    selected_llm_model: str = Form(...),
    selected_embedding_provider: str = Form(...),
    selected_embedding_model: str = Form(...),
    custom_prompt_template: str = Form(None) 
):
    global current_settings, vector_store
    if selected_llm_provider.lower() == "openai":
        if selected_llm_model not in AVAILABLE_OPENAI_LLMS:
            raise HTTPException(status_code=400, detail=f"Invalid OpenAI LLM model: {selected_llm_model}")
    elif selected_llm_provider.lower() == "gemini":
        if selected_llm_model not in AVAILABLE_GEMINI_LLMS:
            raise HTTPException(status_code=400, detail=f"Invalid Gemini LLM model: {selected_llm_model}")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid LLM provider: {selected_llm_provider}")
    api_compat_embed_model = selected_embedding_model
    if selected_embedding_provider.lower() == "openai":
        if selected_embedding_model not in AVAILABLE_OPENAI_EMBEDDING_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid OpenAI Embedding model: {selected_embedding_model}")
    elif selected_embedding_provider.lower() == "gemini":
        if selected_embedding_model == "text-embedding-004" and "models/" not in selected_embedding_model:
            api_compat_embed_model = "models/text-embedding-004"
        if api_compat_embed_model not in AVAILABLE_GEMINI_EMBEDDING_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid Gemini Embedding model: {selected_embedding_model}")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid Embedding provider: {selected_embedding_provider}")
    try:
        get_llm(selected_llm_provider, selected_llm_model)
        get_embeddings(selected_embedding_provider, api_compat_embed_model)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=f"Cannot switch models: {e.detail}")
    current_settings.selected_llm_provider = selected_llm_provider.lower()
    current_settings.selected_llm_model = selected_llm_model
    current_settings.selected_embedding_provider = selected_embedding_provider.lower()
    current_settings.selected_embedding_model = api_compat_embed_model
    if custom_prompt_template is not None:
        # Basic validation for prompt template placeholders
        if "{context}" not in custom_prompt_template or "{question}" not in custom_prompt_template:
            print(f"Warning: Custom prompt template provided during update is missing required placeholders. Original template: '{custom_prompt_template}'. This might lead to issues if not corrected before querying.")
            # Decide if you want to reject, or accept and let the query-time validation handle it.
            # For now, we accept it, and query_rag will fallback.
        current_settings.custom_prompt_template = custom_prompt_template
    print(f"Admin settings updated. Re-initializing vector store with new embedding model: {current_settings.selected_embedding_model}")
    if pc:
        initialize_vector_store() 
    else:
        print("Skipping vector store re-initialization as Pinecone client is not available.")
    print(f"Admin settings updated: LLM: {current_settings.selected_llm_provider}/{current_settings.selected_llm_model}, "
          f"Embedding: {current_settings.selected_embedding_provider}/{current_settings.selected_embedding_model}")
    return {"message": "Admin settings updated successfully."}

# Admin Panel HTML interface
@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    s = current_settings
    # Build options for LLM models
    openai_llms = "".join([f'<option value="{m}" {"selected" if m==s.selected_llm_model else ""}>{m}</option>' for m in AVAILABLE_OPENAI_LLMS])
    gemini_llms = "".join([f'<option value="{m}" {"selected" if m==s.selected_llm_model else ""}>{m}</option>' for m in AVAILABLE_GEMINI_LLMS])
    # Build options for Embedding models
    openai_embs = "".join([f'<option value="{m}" {"selected" if m==s.selected_embedding_model else ""}>{m}</option>' for m in AVAILABLE_OPENAI_EMBEDDING_MODELS.keys()])
    gemini_embs = "".join([f'<option value="{m}" {"selected" if m==s.selected_embedding_model else ""}>{m}</option>' for m in AVAILABLE_GEMINI_EMBEDDING_MODELS.keys()])
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Admin Panel</title>
</head>
<body>
<h1>Admin Panel</h1>
<form method="post" action="/admin/settings/">
<label>LLM Provider:</label>
<select name="selected_llm_provider">
<option value="openai" {"selected" if s.selected_llm_provider=="openai" else ""}>OpenAI</option>
<option value="gemini" {"selected" if s.selected_llm_provider=="gemini" else ""}>Gemini</option>
</select><br/><br/>
<label>LLM Model:</label>
<select name="selected_llm_model">
<optgroup label="OpenAI">{openai_llms}</optgroup>
<optgroup label="Gemini">{gemini_llms}</optgroup>
</select><br/><br/>
<label>Embedding Provider:</label>
<select name="selected_embedding_provider">
<option value="openai" {"selected" if s.selected_embedding_provider=="openai" else ""}>OpenAI</option>
<option value="gemini" {"selected" if s.selected_embedding_provider=="gemini" else ""}>Gemini</option>
</select><br/><br/>
<label>Embedding Model:</label>
<select name="selected_embedding_model">
<optgroup label="OpenAI">{openai_embs}</optgroup>
<optgroup label="Gemini">{gemini_embs}</optgroup>
</select><br/><br/>
<label>Custom Prompt Template:</label><br/>
<textarea name="custom_prompt_template" rows="10" cols="80">{s.custom_prompt_template}</textarea><br/><br/>
<button type="submit">Save Settings</button>
</form>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/")
async def read_root():
    return {"message": "RAG Chatbot Backend is running. Navigate to /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    if not PINECONE_INDEX_NAME:
        print("Error: PINECONE_INDEX_NAME is not set. Please define it.")
    else:
        print(f"Attempting to start Uvicorn server for RAG chatbot on index: {PINECONE_INDEX_NAME}")
        uvicorn.run("app:app", host="0.0.0.0", port=8000)