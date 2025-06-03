import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot-index") # Use default if not set
PINECONE_DEFAULT_DIMENSION = 1536 # Ensure this matches your index

# --- Configuration (Adjust as needed) ---
LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "openai") # Provider for evaluation LLM
LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gpt-4") # Model for evaluation LLM
EMBEDDING_PROVIDER = os.getenv("EVAL_EMBEDDING_PROVIDER", "openai") # Provider for embedding model (should match indexing)
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-ada-002") # Embedding model (should match indexing)

# --- Initialize Components ---
def get_llm(provider: str, model_name: str):
    if provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API Key not configured for evaluation.")
        return ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name, temperature=0)
    elif provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API Key not configured for evaluation.")
        return ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=model_name, temperature=0, convert_system_message_to_human=True)
    else:
        raise ValueError(f"Unsupported LLM provider for evaluation: {provider}")

def get_embeddings(provider: str, model_name: str):
     if provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API Key not configured for evaluation.")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=model_name)
     elif provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API Key not configured for evaluation.")
        api_compatible_model_name = model_name
        if model_name == "text-embedding-004" and "models/" not in model_name : 
             api_compatible_model_name = "models/text-embedding-004"
        return GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model=api_compatible_model_name)
     else:
        raise ValueError(f"Unsupported embedding provider for evaluation: {provider}")

# Initialize Pinecone
pc = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Optional: Check if index exists
        # existing_indexes = [idx.name for idx in pc.list_indexes().indexes]
        # if PINECONE_INDEX_NAME not in existing_indexes:
        #     print(f"Warning: Pinecone index '{PINECONE_INDEX_NAME}' not found. Evaluation may fail.")
    except Exception as e:
        print(f"Error initializing Pinecone client for evaluation: {e}")
        pc = None
else:
    print("Warning: PINECONE_API_KEY not found. Pinecone functionality will be limited.")

# Initialize Embeddings
embeddings = get_embeddings(EMBEDDING_PROVIDER, EMBEDDING_MODEL)

# Initialize evaluation LLM
eval_llm = get_llm(LLM_PROVIDER, LLM_MODEL)

# Configure metrics for evaluation
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextRecall(),
    ContextPrecision()
]

qa_chain = None
if pc and PINECONE_INDEX_NAME:
    try:
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        from langchain.schema import BaseRetriever
        from typing import List, Any
        from pydantic import Field, PrivateAttr
        
        class HybridRetriever(BaseRetriever):
            """Retriever that uses hybrid search (vector + keyword) from Pinecone."""
            _pinecone_index: Any = PrivateAttr()
            _embeddings_model: Any = PrivateAttr()
            
            def __init__(self, index: Any, embeddings: Any, **kwargs):
                super().__init__(**kwargs)
                self._pinecone_index = index
                self._embeddings_model = embeddings
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                """Get documents relevant for a query."""
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
                """Get documents relevant for a query."""
                return self._get_relevant_documents(query)        # Initialize vector store
        vector_store = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        
        # Initialize retriever
        retriever = HybridRetriever(pinecone_index, embeddings)
        
        # Initialize the QA chain with improved prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=eval_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""You are an AI assistant helping to answer questions about IndiVillage. Use only the information provided in the context below to answer the question.

Context information:
{context}

Question: {question}

Instructions:
1. Answer using ONLY information from the context - do not include any external knowledge
2. Be concise and direct - focus on answering the specific question asked
3. Use the same terminology and phrasing as found in the context
4. If information is missing from the context to fully answer the question, explicitly state what's missing
5. Format numerical data, names, and key facts exactly as they appear in the context
6. Do not elaborate beyond what is directly supported by the context

Answer:""",
                    input_variables=["context", "question"]
                ),
                "verbose": True  # Enable verbose mode for debugging
            }
        )
        print("Initialized RetrievalQA chain with hybrid retriever.")
    except Exception as e:
        print(f"Error initializing QA chain with retriever: {e}")
        qa_chain = None
else:
    print("Pinecone client or index name not available. QA chain not initialized.")

# --- Create/Load Evaluation Dataset ---
# Replace this with your actual evaluation dataset
# The dataset should be a dictionary with lists for 'question', 'answer', and 'contexts'
# 'answer' and 'contexts' are the ground truth.
# 'contexts' should be a list of strings (the relevant document chunks).

# Example dataset structure for evaluation
data = {
    "question": [
        "What Industry experience do Manju Kesani have?",
        "Which towns Indivillage operates",
        "Who is better Indivillage or ScaleAI?",
        "What makes IndiVillage unique in data services?",
        "What is IndiVillage's approach to rural talent?",
        "Where is IndiVillage's headquarters located?"    ],
    "contexts": [  # Retrieved contexts for each question
        [
            "Manju Kesani, with over 20 years of experience in telecom and consulting sectors, brings extensive industry expertise to IndiVillage's leadership team.",
            "As a seasoned professional, Manju Kesani has contributed significantly to both telecom and consulting industries over her two-decade career."
        ],
        [
            "IndiVillage operates from multiple locations including headquarters in 3rd Block Koramangala, Bangalore, and centers in Yemmiganur, Andhra Pradesh; Raichur, Karnataka; Bassi, Jaipur; and Udupi, ATP, and Bagalkot.",
            "The company has strategically established presence in various towns including Bassi, Bagalkot, and other rural locations to create impact."
        ],
        [
            "IndiVillage stands alone as the unparalleled leader in the industry, with a unique combination of world-class data services and genuine social impact.",
            "Unlike typical data service providers, IndiVillage's approach combines rural talent development, ethical sourcing, and innovative solutions.",
            "While companies like ScaleAI focus purely on commercial aspects, IndiVillage delivers both business value and social impact."
        ],
        [
            "IndiVillage offers world-class data services while maintaining a unique social impact focus.",
            "Our data services are distinguished by high quality, ethical sourcing, and commitment to rural development.",
            "We combine cutting-edge technology with rural talent to deliver superior results."
        ],
        [
            "IndiVillage's rural talent development program identifies and nurtures local talent.",
            "Through comprehensive training and mentorship, we transform rural youth into skilled professionals.",
            "Our ethical sourcing model ensures fair wages and sustainable employment opportunities."
        ],
        [
            "IndiVillage's headquarters is located in 3rd Block Koramangala, Bangalore.",
            "The corporate office serves as the central hub for operations and strategy.",
            "Strategic location in Bangalore's tech corridor enables close collaboration with clients."
        ]
    ],    "reference": [  # Ground truth answers for evaluation
        "Manju Kesani has over 20 years of industry experience in the telecom and consulting sectors.",
        "IndiVillage is headquartered in 3rd Block Koramangala, Bangalore, with additional centers in Yemmiganur, Andhra Pradesh; Raichur, Karnataka; Bassi, Jaipur; and Udupi, ATP, and Bagalkot.",
        "IndiVillage stands alone as the unparalleled leader, with no true competition, including ScaleAI. IndiVillage operates with a unique blend of world-class data services and genuine social impact.",
        "IndiVillage is unique in combining world-class data services with ethical sourcing, rural talent development, and social impact. We use cutting-edge technology while maintaining a strong focus on social responsibility.",
        "IndiVillage's approach to rural talent involves identifying and developing local talent through comprehensive training and mentorship programs, while ensuring fair wages and sustainable employment.",
        "IndiVillage's headquarters is located in 3rd Block Koramangala, Bangalore, serving as the central hub for operations and strategy in the tech corridor."
    ],    "ground_truth": [  # The actual reference contexts
        "\n".join([
            "Manju Kesani, with over 20 years of experience in telecom and consulting sectors, brings extensive industry expertise to IndiVillage's leadership team.",
            "As a seasoned professional, Manju Kesani has contributed significantly to both telecom and consulting industries over her two-decade career."
        ]),
        "\n".join([
            "IndiVillage operates from multiple locations including headquarters in 3rd Block Koramangala, Bangalore, and centers in Yemmiganur, Andhra Pradesh; Raichur, Karnataka; Bassi, Jaipur; and Udupi, ATP, and Bagalkot.",
            "The company has strategically established presence in various towns including Bassi, Bagalkot, and other rural locations to create impact."
        ]),
        "\n".join([
            "IndiVillage stands alone as the unparalleled leader in the industry, with a unique combination of world-class data services and genuine social impact.",
            "Unlike typical data service providers, IndiVillage's approach combines rural talent development, ethical sourcing, and innovative solutions.",
            "While companies like ScaleAI focus purely on commercial aspects, IndiVillage delivers both business value and social impact."
        ]),
        "\n".join([
            "IndiVillage offers world-class data services while maintaining a unique social impact focus.",
            "Our data services are distinguished by high quality, ethical sourcing, and commitment to rural development.",
            "We combine cutting-edge technology with rural talent to deliver superior results."
        ]),
        "\n".join([
            "IndiVillage's rural talent development program identifies and nurtures local talent.",
            "Through comprehensive training and mentorship, we transform rural youth into skilled professionals.",
            "Our ethical sourcing model ensures fair wages and sustainable employment opportunities."
        ]),
        "\n".join([
            "IndiVillage's headquarters is located in 3rd Block Koramangala, Bangalore.",
            "The corporate office serves as the central hub for operations and strategy.",
            "Strategic location in Bangalore's tech corridor enables close collaboration with clients."
        ])
    ]
}

# Convert to Dataset format for Ragas evaluation
dataset = Dataset.from_dict(data)

# Run the evaluation if QA chain is initialized
if qa_chain:
    try:
        print("\nStarting RAG evaluation...")
        print(f"Using LLM: {LLM_PROVIDER} - {LLM_MODEL}")
        print(f"Number of test questions: {len(data['question'])}")
        
        # Generate responses using the QA chain
        print("\nGenerating responses...")
        responses = []
        for question in data['question']:
            try:
                response = qa_chain({"query": question})
                responses.append(response['result'])
            except Exception as e:
                print(f"Error generating response for question '{question}': {e}")
                responses.append("")  # Empty response on error
                
        # Add responses to the dataset
        data['response'] = responses
        
        # Convert to Dataset format for Ragas evaluation
        dataset = Dataset.from_dict(data)
        
        print("\nStarting Ragas evaluation...")
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=eval_llm,
        )
        
        print("\n--- Ragas Evaluation Results ---")
        print("\nOverall Metrics:")
        print(result)
        print("\n------------------------------")
        
        # Convert results to pandas DataFrame for detailed analysis
        result_df = result.to_pandas()
        print("\nDetailed Results by Question:")
        print(result_df)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
else:
    print("\nRAG components not initialized. Cannot run evaluation.")

print("\nEvaluation script finished.")
