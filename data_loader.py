import os
from datasets import load_dataset

def download_and_prepare_data():
    dataset = load_dataset("PatronusAI/financebench", split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    docs = []
    for i, item in enumerate(dataset):
        # Extract evidence texts (list of dicts)
        evidence_list = item.get('evidence', [])
        evidence_texts = " ".join(evi.get('evidence_text', '') for evi in evidence_list)
        
        docs.append({
            "doc_id": str(i),
            "company": item.get('company', ''),
            "doc_name": item.get('doc_name', ''),
            "content": evidence_texts,
            "question": item.get('question', ''),
            "answer": item.get('answer', ''),
            "justification": item.get('justification', '')
        })
    
    # Print one sample for verification
    print("\nSample Processed Document:\n", docs[0])
    
    return docs

# Run and verify
if __name__ == "__main__":
    docs = download_and_prepare_data()
