import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings')
SIMILARITY_SCORE_LIMIT = 0.9
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_NAME = "phamhai/Llama-3.2-3B-Instruct-Frog"


def product_suggestion_api(data):
    vectorstore = load_faiss_embeddings()
    query_str = extract_user_query(data['query'])
    matches = re.findall(r'\{\s*"formatted_query"\s*:\s*"(.*?)"\s*\}', query_str)

    formatted_query = matches[-1] if matches else ""
    print("Model Formatted Query:", formatted_query)

    results = []
    docs = vectorstore.similarity_search_with_score(formatted_query, k=4)

    for doc, score in docs:
        print("Score of suggestion:", score)
        if score <= SIMILARITY_SCORE_LIMIT:
            result = extract_values(doc, formatted_query)
            if result:
                results.append(result)

    return results


def load_faiss_embeddings():
    with open(EMBEDDINGS_PATH, 'rb') as f:
        serialized = f.read()

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    return FAISS.deserialize_from_bytes(
        serialized,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def extract_user_query(query):
    prompt = f"""
User has a medical Query. Query may also be very contextually large.
Your job is to extract one line query. If query is already brief return the same query.
For example:

Example User Query - i have body pain from 10 days and itching also  
Expected response - body pain and itching from 10 days  

Response should only be in String in mentioned format:
{{
    "formatted_query": "extracted query"
}}

User query -
{query}
    """
    messages = [
        {"role": "assistant", "content": "You are an assistant with deep knowledge about medical tests"},
        {"role": "user", "content": prompt}
    ]
    return send_prompt(messages)


def send_prompt(messages, temperature=0.2):
    formatted = format_messages(messages)
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16, device_map="auto")

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def format_messages(messages):
    prompt = ""
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        prompt += f"[{role}]\n{content}\n"
    return prompt + "[Assistant]\n"


def extract_values(doc, query):
    text = doc.page_content
    patterns = {
        'adr': r"ADR:\s*(.*?)\s*(?=Drug:|Disease:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)",
        'drug': r"Drug:\s*(.*?)\s*(?=ADR:|Disease:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)",
        'disease': r"Disease:\s*(.*?)\s*(?=ADR:|Drug:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)",
        'symptom': r"Symptom:\s*(.*?)\s*(?=ADR:|Drug:|Disease:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)",
        'other_drugs': r"Other Drugs:\s*(.*?)\s*(?=ADR:|Drug:|Disease:|Symptom:|File Path:|metadata=|$)"
    }

    extracted = {key: extract_pattern(pattern, text) for key, pattern in patterns.items()}
    if extracted["disease"] or extracted["symptom"]:
        return extracted
    return {}


def extract_pattern(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return [x.strip() for x in content.split(',')] if content else []
    return []


def data_prep_in_json(data):
    merged = {
        'Drug': set(),
        'Symptoms/Diseases': set(),
        'ADEs': set()
    }

    for entry in data:
        merged['ADEs'].update(entry.get('adr', []))
        merged['Drug'].update(entry.get('drug', []))
        merged['Drug'].update(entry.get('other_drugs', []))
        merged['Symptoms/Diseases'].update(entry.get('symptom', []))
        merged['Symptoms/Diseases'].update(entry.get('disease', []))

    final_result = {key: sorted(list(value)) for key, value in merged.items()}
    json_output = json.dumps(final_result, indent=4)

    output_path = os.path.join(BASE_DIR, 'hugging_face_suggestion.json')
    with open(output_path, "w") as f:
        f.write(json_output)

    print('Suggested Drugs, Symptoms/Diseases, ADEs stored in: hugging_face_suggestion.json')
    return json_output


def main():
    print('\nAsk your medical query: ')
    query = input().strip()
    word_count = len(query.split())

    if word_count < 3:
        print("Please provide more details.")
        return

    global SIMILARITY_SCORE_LIMIT
    for attempt in range(1, 4):
        print(f"\nAttempt: {attempt}")
        response = product_suggestion_api({"query": query})

        if response:
            data_prep_in_json(response)
            return

        print("No Response")
        SIMILARITY_SCORE_LIMIT += 0.2
        print(f"Increased similarity score limit to: {SIMILARITY_SCORE_LIMIT}")

    print("Please rephrase your query for better results.")


if __name__ == "__main__":
    main()