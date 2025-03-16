import os
import time
import wandb
from dotenv import load_dotenv
from pinecone import Pinecone
from cohere_emb import Client as CohereClient
from voyageai import Client as VoyageClient
from groq import Client as GroqClient
from datasets import load_dataset
from evaluation import compute_metrics

# Athina Logger Imports
from athina_logger.api_key import AthinaApiKey
from athina_logger.inference_logger import InferenceLogger
from athina_logger.exception.custom_exception import CustomException

# ------------------- Load environment variables -------------------
load_dotenv()

# ------------------- Initialize W&B -------------------
wandb.init(project='rag_pipeline_monitoring')

# ------------------- Initialize Pinecone -------------------
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
cohere_index = pc.Index('cohere-financebench-index')
voyage_index = pc.Index('voyage-financebench-index')

# ------------------- Initialize Embedding Clients -------------------
cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
voyage_client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))

# ------------------- Initialize Groq Client -------------------
groq_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

# ------------------- Initialize Athina API Key -------------------
AthinaApiKey.set_api_key(os.getenv('ATHINA_API_KEY'))

# ------------------- Embedding Functions -------------------
def get_cohere_embedding(text):
    response = cohere_client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings[0]

def get_voyage_embedding(text):
    response = voyage_client.embed(
        texts=[text],
        model="voyage-02"
    )
    return response.embeddings[0]

# ------------------- Retrieval Function -------------------
def retrieve_docs(query, model="cohere"):
    start_time = time.time()
    if model == "cohere":
        print("Using Cohere Embeddings for Retrieval")
        embed_query = get_cohere_embedding(query)
        res = cohere_index.query(vector=embed_query, top_k=10, include_metadata=True)
    elif model == "voyage":
        print("Using Voyage AI Embeddings for Retrieval")
        embed_query = get_voyage_embedding(query)
        res = voyage_index.query(vector=embed_query, top_k=10, include_metadata=True)
        # -------------- Sleep to avoid Voyage RPM limit --------------
        print("Sleeping 20 seconds to avoid Voyage AI rate limit...")
        time.sleep(20)
    else:
        raise ValueError("Model must be 'cohere' or 'voyage'")
    retrieval_time = time.time() - start_time
    wandb.log({f"{model}_retrieval_time": retrieval_time})
    return res, embed_query

# ------------------- Answer Synthesis Function -------------------
def synthesize_answer(query, retrieved_docs):
    context = "\n\n".join([doc['metadata']['content'] for doc in retrieved_docs['matches']])
    prompt = f"""
You are a financial analyst. Use only the provided context to answer the question concisely, factually, and in well-structured sentences.

Context:
{context}

Question:
{query}

Answer:
"""
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable financial expert."},
            {"role": "user", "content": prompt}
        ],
        model="mixtral-8x7b-32768",
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    return answer

# ------------------- Metric Summarization -------------------
def summarize_metrics(results):
    total = {"EM": 0, "F1": 0, "BLEU": 0, "ROUGE": 0}
    for _, metrics in results:
        for key in total:
            total[key] += metrics[key]
    count = len(results)
    avg = {key: round(total[key] / count, 4) for key in total}
    return avg

# ------------------- Main Pipeline -------------------
def main():
    dataset = load_dataset("PatronusAI/financebench", split="train[:20]")  # First 20 samples

    cohere_results = []
    voyage_results = []

    for idx, sample in enumerate(dataset):
        question = sample['question']
        ground_truth = sample['answer']

        # ------------------ Cohere Retrieval ------------------
        cohere_docs, _ = retrieve_docs(question, model="cohere")
        cohere_answer = synthesize_answer(question, cohere_docs)
        cohere_metrics = compute_metrics(cohere_answer, ground_truth)

        # Log to W&B
        wandb.log({f"cohere_sample_{idx}_metrics": cohere_metrics})

        # Log to Athina
        try:
            InferenceLogger.log_inference(
                prompt_slug="financebench_cohere",
                prompt={"question": question},
                language_model_id="mixtral-8x7b-32768",
                response=cohere_answer,  # pass string
                external_reference_id=f"cohere_{idx}",
                custom_attributes={
                    "model": "cohere",
                    "index": idx,
                    "ground_truth": ground_truth,
                    **cohere_metrics
                }
            )
        except CustomException as e:
            print(f"Athina Logging Error (Cohere): {e.status_code} - {e.message}")

        cohere_results.append((cohere_answer, cohere_metrics))

        # ------------------ Voyage Retrieval ------------------
        voyage_docs, _ = retrieve_docs(question, model="voyage")
        voyage_answer = synthesize_answer(question, voyage_docs)
        voyage_metrics = compute_metrics(voyage_answer, ground_truth)

        # Log to W&B
        wandb.log({f"voyage_sample_{idx}_metrics": voyage_metrics})

        # Log to Athina
        try:
            InferenceLogger.log_inference(
                prompt_slug="financebench_voyage",
                prompt={"question": question},
                language_model_id="mixtral-8x7b-32768",
                response=voyage_answer,  # pass string
                external_reference_id=f"voyage_{idx}",
                custom_attributes={
                    "model": "voyage",
                    "index": idx,
                    "ground_truth": ground_truth,
                    **voyage_metrics
                }
            )
        except CustomException as e:
            print(f"Athina Logging Error (Voyage): {e.status_code} - {e.message}")

        voyage_results.append((voyage_answer, voyage_metrics))

        # Print progress
        print(f"\nSample {idx+1}:")
        print(f"Question: {question}")
        print(f"Cohere Answer: {cohere_answer}")
        print(f"Voyage Answer: {voyage_answer}")

    # ------------------ Summary & Comparison ------------------
    print("\n=== Cohere Average Metrics ===")
    avg_cohere = summarize_metrics(cohere_results)
    print(avg_cohere)

    print("\n=== Voyage Average Metrics ===")
    avg_voyage = summarize_metrics(voyage_results)
    print(avg_voyage)

    wandb.log({
        "cohere_avg_metrics": avg_cohere,
        "voyage_avg_metrics": avg_voyage
    })

    wandb.finish()

# ------------------- Run -------------------
if __name__ == "__main__":
    main()
