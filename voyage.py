import os
from datasets import load_dataset
from voyageai import Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Voyage client
voyage_client = Client(api_key=os.getenv('VOYAGE_API_KEY'))

# ------------------- Download FinanceBench data -------------------
def download_and_prepare_data():
    dataset = load_dataset("PatronusAI/financebench", split="train")
    docs = []
    for i, item in enumerate(dataset):
        docs.append({
            "doc_id": str(i),
            "company": item.get('company', 'N/A'),
            "doc_name": item.get('doc_name', 'N/A'),
            "content": item.get('content', ''),
            "question": item.get('question', ''),
            "answer": item.get('answer', ''),
            "justification": item.get('justification', '')
        })
    return docs

# Download and prepare
docs = download_and_prepare_data()

# Extract text contents for embedding
texts = [f"{doc['content']} {doc['question']} {doc['answer']} {doc['justification']}" for doc in docs]

# ------------------- Generate embeddings using Voyage AI -------------------
print("Generating Voyage AI embeddings...")

batch_size = 2  
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1)//batch_size + 1}...")

    try:
        response = voyage_client.embed(
            texts=batch_texts,   
            model="voyage-02"
        )
        embeddings = response.embeddings
        all_embeddings.extend(embeddings)

        
        time.sleep(21)
    
    except Exception as e:
        print(f"Error on batch {i // batch_size + 1}: {e}")
        print("Sleeping 60s before retrying...")
        time.sleep(10)
        continue

# Validate
assert len(all_embeddings) == len(docs), "Mismatch between embeddings and documents!"

# ------------------- Initialize Pinecone -------------------
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define index
index_name = 'voyage-financebench-index'
dimension = len(all_embeddings[0])  

# Delete if exists
if index_name in pc.list_indexes().names():
    print(f"Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)

# Create index
print(f"Creating Pinecone index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Connect
index = pc.Index(index_name)

# ------------------- Upsert to Pinecone -------------------
print("Upserting data to Pinecone...")
for i, doc in enumerate(docs):
    metadata = {
        'company': str(doc.get('company', '') or ''),
        'doc_name': str(doc.get('doc_name', '') or ''),
        'question': str(doc.get('question', '') or ''),
        'answer': str(doc.get('answer', '') or ''),
        'justification': str(doc.get('justification', '') or ''),
        'content': str(doc.get('content', '') or '')
    }
    index.upsert([(str(i), all_embeddings[i], metadata)])

print("FinanceBench data preparation and indexing complete using Voyage AI!")
