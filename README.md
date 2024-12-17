# chat-with-pdf
import os
from typing import List
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai

# Step 1: Data Ingestion
class PDFProcessor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

  def extract_text(self, pdf_path: str) -> str:
       # """Extract text from a single PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

   def process_pdfs(self) -> List[str]:
        #"""Extract and chunk text from all PDFs in the directory."""
        all_text = []
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                text = self.extract_text(pdf_path)
                all_text.append(text)
        return all_text

# Step 2: Embedding and Storing
class EmbeddingStore:
    def __init__(self, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_store = None

   def create_embeddings(self, texts: List[str]):
        #"""Create vector embeddings and store them in FAISS."""
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = FAISS.from_texts(texts, embeddings)

   def save_vector_store(self, path: str):
       # """Save the FAISS vector store to a file."""
        self.vector_store.save_local(path)

   def load_vector_store(self, path: str):
        #"""Load the FAISS vector store from a file."""
        self.vector_store = FAISS.load_local(path, embeddings=self.embedding_model)

# Step 3: Query Handling
class QueryProcessor:
    def __init__(self, vector_store_path: str):
        self.vector_store = FAISS.load_local(vector_store_path, embeddings=HuggingFaceEmbeddings())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the 'OPENAI_API_KEY' environment variable.")
        openai.api_key = api_key
        self.llm = OpenAI(temperature=0, model="gpt-4")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.vector_store.as_retriever()
        )

   def handle_query(self, query: str) -> str:
       # """Process the query and return an LLM-generated response."""
        return self.qa_chain.run(query)

# Step 4: Comparison Handling
class ComparisonProcessor(QueryProcessor):
    def handle_comparison_query(self, query: str) -> str:
       # """Handle queries that require comparison across multiple documents."""
        # This can be extended for domain-specific comparisons
        return self.handle_query(query)

# Step 5: Integration
if __name__ == "__main__":
    pdf_directory = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables"
    vector_store_path = "./vector_store"

    # Step 1: Process PDFs
  pdf_processor = PDFProcessor(pdf_directory)
    texts = pdf_processor.process_pdfs()

    # Step 2: Embed and store
   embedding_store = EmbeddingStore("sentence-transformers/all-MiniLM-L6-v2")
    embedding_store.create_embeddings(texts)
    embedding_store.save_vector_store(vector_store_path)

    # Step 3 and 4: Query handling
  
  query_processor = QueryProcessor(vector_store_path)

  while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = query_processor.handle_query(user_query)
        print("Response:", response)
