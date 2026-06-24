import os
from typing import List
import json
import argparse
import logging

from src.semflowrag import SemFlowRAG

def main():

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/local_test'  # Define save directory for SemFlowRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Any OpenAI model name
    embedding_model_name = 'nvidia/NV-Embed-v2'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a SemFlowRAG instance
    semflowrag = SemFlowRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        llm_base_url="http://localhost:6578/v1"
                        )

    # Run indexing
    semflowrag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print(semflowrag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a SemFlowRAG instance
    semflowrag = SemFlowRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        azure_endpoint="https://bernal-semflowrag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                        azure_embedding_endpoint="https://bernal-semflowrag.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
                        )

    print(semflowrag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a SemFlowRAG instance
    semflowrag = SemFlowRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        azure_endpoint="https://bernal-semflowrag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                        azure_embedding_endpoint="https://bernal-semflowrag.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
                        )

    new_docs = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."]

    # Run indexing
    semflowrag.index(docs=new_docs)

    print(semflowrag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."
    ]

    semflowrag.delete(docs_to_delete)

    print(semflowrag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

if __name__ == "__main__":
    main()
