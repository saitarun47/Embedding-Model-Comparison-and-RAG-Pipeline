from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    return {
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "voyage_api_key": os.getenv("VOYAGE_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "athina_api_key": os.getenv("ATHINA_API_KEY"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }
