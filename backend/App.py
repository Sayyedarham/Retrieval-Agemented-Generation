
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os
import tempfile
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_DB_PATH = 'vectordb'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize vector store
vectorstore = None
documents_corpus = []  # For BM25
document_chunks = []  # Store chunks with metadata

# Initialize LLM (using Ollama - you can switch to OpenAI/Anthropic)
llm = Ollama(model="tinyllama")  # or use "mistral", "codellama"

def load_document(file_path):
    """Load document based on file type"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    elif ext in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader.load()

def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def rewrite_query(query):
    """Rewrite query for better retrieval using LLM"""
    rewrite_prompt = f"""Given the following user query, rewrite it to be more specific and detailed for better document retrieval. 
    Expand abbreviations, add relevant keywords, and make it more search-friendly.
    
    Original query: {query}
    
    Rewritten query (return ONLY the rewritten query, nothing else):"""
    
    try:
        rewritten = llm(rewrite_prompt).strip()
        return rewritten
    except:
        return query  # Fallback to original if rewriting fails

def hybrid_search(query, k=5):
    """Perform hybrid search: semantic (vector) + keyword (BM25)"""
    global vectorstore, documents_corpus, document_chunks
    
    if vectorstore is None:
        return []
    
    # Semantic search
    semantic_results = vectorstore.similarity_search_with_score(query, k=k)
    
    # BM25 keyword search
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(documents_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top k indices from BM25
    top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
    
    # Combine results with weighted scoring
    combined_results = {}
    
    # Add semantic results (weight: 0.6)
    for doc, score in semantic_results:
        doc_id = doc.metadata.get('source', '') + str(doc.metadata.get('page', 0))
        combined_results[doc_id] = {
            'document': doc,
            'score': (1 - score) * 0.6,  # Convert distance to similarity
            'type': 'semantic'
        }
    
    # Add BM25 results (weight: 0.4)
    for idx in top_bm25_indices:
        if idx < len(document_chunks):
            doc = document_chunks[idx]
            doc_id = doc.metadata.get('source', '') + str(doc.metadata.get('page', 0))
            
            if doc_id in combined_results:
                combined_results[doc_id]['score'] += bm25_scores[idx] * 0.4
                combined_results[doc_id]['type'] = 'hybrid'
            else:
                combined_results[doc_id] = {
                    'document': doc,
                    'score': bm25_scores[idx] * 0.4,
                    'type': 'keyword'
                }
    
    # Sort by combined score
    sorted_results = sorted(
        combined_results.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:k]
    
    return [(r['document'], r['score']) for r in sorted_results]

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents"""
    global vectorstore, documents_corpus, document_chunks
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    processed_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Save file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            # Load and process document
            documents = load_document(file_path)
            chunks = chunk_documents(documents)
            
            # Update document corpus for BM25
            for chunk in chunks:
                documents_corpus.append(chunk.page_content.lower().split())
                document_chunks.append(chunk)
            
            # Add to vector store
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=VECTOR_DB_PATH
                )
            else:
                vectorstore.add_documents(chunks)
            
            vectorstore.persist()
            
            processed_files.append({
                'name': file.filename,
                'size': os.path.getsize(file_path),
                'chunks': len(chunks)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing {file.filename}: {str(e)}'}), 500
    
    return jsonify({
        'message': 'Documents processed successfully',
        'files': processed_files
    })

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Query documents with hybrid search and query rewriting"""
    global vectorstore
    
    if vectorstore is None:
        return jsonify({'error': 'No documents uploaded yet'}), 400
    
    data = request.json
    original_query = data.get('query', '')
    
    if not original_query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Rewrite query
    rewritten_query = rewrite_query(original_query)
    
    # Perform hybrid search
    search_results = hybrid_search(rewritten_query, k=5)
    
    # Create context from top results
    context = "\n\n".join([doc.page_content for doc, _ in search_results[:3]])
    
    # Generate answer using LLM
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always cite the specific information from the context.

    Context:
    {context}

    Question: {question}

    Detailed Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    final_prompt = prompt.format(context=context, question=rewritten_query)
    answer = llm(final_prompt)
    
    # Format sources with relevance scores
    sources = []
    for doc, score in search_results:
        sources.append({
            'document': os.path.basename(doc.metadata.get('source', 'Unknown')),
            'page': doc.metadata.get('page', 0),
            'relevance': float(score),
            'excerpt': doc.page_content[:200] + "..."
        })
    
    return jsonify({
        'answer': answer,
        'original_query': original_query,
        'rewritten_query': rewritten_query,
        'search_type': 'Hybrid (Semantic + Keyword)',
        'sources': sources
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(document_chunks),
        'vectorstore_initialized': vectorstore is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
