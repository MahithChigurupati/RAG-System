import os
import glob
import fitz
import cohere
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

def load_environment():
    """Load environment variables and API keys"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    print("API Key Loaded Successfully" if api_key else "API Key Not Found")
    return api_key, cohere_api_key

def load_all_pdfs_in_directory(directory="./content/"):
    """Load and combine text from all PDFs in the specified directory"""
    combined_text = ""
    pdf_paths = glob.glob(f"{directory}/*.pdf")

    for file_path in pdf_paths:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                combined_text += page.get_text()

    return combined_text

def split_text(text):
    """Split text into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    return text_splitter.split_text(text)

def create_embeddings_and_index(documents):
    """Create embeddings and FAISS index"""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index = FAISS.from_texts(texts=documents, embedding=embeddings)
    faiss_index.save_local("faiss_index")
    return embeddings, faiss_index

def setup_retriever(vectorstore):
    """Set up the base retriever"""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def setup_compression_retriever(retriever, cohere_api_key):
    """Set up the compression retriever with Cohere reranking"""
    cohere_client = cohere.ClientV2(cohere_api_key)
    cohere_reranker = CohereRerank(client=cohere_client, model="rerank-english-v3.0", top_n=3)
    
    return ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=retriever
    )

def setup_qa_chain(compression_retriever):
    """Set up the QA chain"""
    llm = ChatOpenAI(model="gpt-4-0613", temperature=0.5, max_tokens=3000)
    return RetrievalQA.from_chain_type(llm, retriever=compression_retriever)

def pretty_print(docs):
    """Print documents in a formatted way"""
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def process_queries(qa_chain, queries):
    """Process a list of queries and get responses"""
    for query in queries:
        prompt = (
            "You are an expert assistant with a strong grasp of the subject matter. "
            "Please answer the following question succinctly, highlighting the key points. "
            f"Format your response as follows: \n\n"
            f" [Your answer here]\n"
            f"Key Points: \n"
            f"- Point 1: [Key insight 1]\n"
            f"- Point 2: [Key insight 2]\n"
            f"- Point 3: [Key insight 3]\n\n"
            f"Ensure your response is relevant and avoid unnecessary elaboration. "
            f"Answer the following question: '{query}'"
        )
        response = qa_chain.invoke(prompt)
        print(f"Question: {query}\nAnswer: {response['result']}\n")
        print("-" * 100)

def main():
    # Load environment variables
    api_key, cohere_api_key = load_environment()
    
    # Load and process PDFs
    pdf_data = load_all_pdfs_in_directory()
    print(f"Total characters in PDF data: {len(pdf_data)}")
    
    # Split text into documents
    documents = split_text(pdf_data)
    print(f"Number of documents: {len(documents)}")
    
    # Create embeddings and index
    embeddings, faiss_index = create_embeddings_and_index(documents)
    
    # Load vectorstore
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"Number of vectors in vectorstore: {vectorstore.index.ntotal}")
    print(f"Dimensionality of vectors: {vectorstore.index.d}")
    
    # Set up retrievers
    retriever = setup_retriever(vectorstore)
    compression_retriever = setup_compression_retriever(retriever, cohere_api_key)
    
    # Set up QA chain
    qa_chain = setup_qa_chain(compression_retriever)
    
    # Example queries
    queries = [
        "who does the document discuss about?",
        "what are his achievements?",
        "what certifications does he have?",
        "what are his skills?",
        "Does he have any experience working with AI/ML or Blockchain?",
        "what are his qualifications?",
        "what are his experiences?"
    ]
    
    # Process queries
    process_queries(qa_chain, queries)

if __name__ == "__main__":
    main()