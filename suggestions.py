
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

similarity_score_limit = 0.9

def product_suggestion_api(data):
    deserialize_vector = get_faiss_embeddings()

    query_str = extract_user_query(data['query'])
    matches = re.findall(r'\{\s*"formatted_query"\s*:\s*"(.*?)"\s*\}', query_str)
    
    formatted_query_value = ''
    if matches:
        formatted_query_value = matches[-1]
        print("Model Fomatted Query: ", formatted_query_value)
    
    docs = deserialize_vector.similarity_search_with_score(formatted_query_value, k=4)

    results = []
    for document in docs:
        score = document[1]
        print('score of suggestion: ',  score)

        if score <= similarity_score_limit:
            result = extract_values(document[0], formatted_query_value)
            if result:
                results.append(result)
    return results

def get_faiss_embeddings():

    data_embeddings = os.path.join(base_dir, 'embeddings')

    with open(data_embeddings, 'rb') as f:
        serialize_vector = f.read()
                
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hugging_face_for_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    deserialize_vector = FAISS.deserialize_from_bytes(serialize_vector,hugging_face_for_embeddings,allow_dangerous_deserialization=True)
    return deserialize_vector

def extract_user_query(query):
    prompt = """
    User has a medical Query. Query may also be very contextually large.
    Your job is to extract one line query. If query is already brief return the same query
    For example

    example User Query - i have body pain from 10 days and itching also
    Expected response - body pain and itching from 10 days 
    
    Response should only be in String in mentioned format
    {
        "formatted_query" : "extracted query"
    }
    
    User query -
    """+query
    gpt_prompt_input = [{"role": "assistant", "content": "You are an assistant with deep knowledge about medical tests"},
                        {"role": "user", "content": prompt}]

    data = send_prompt(gpt_prompt_input)
    # data = json.loads(completion)
    return data

def send_prompt(prompt_list, temperature=0.2):

    formatted_prompt = format_messages(prompt_list)

    model_name = "phamhai/Llama-3.2-3B-Instruct-Frog"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def format_messages(messages):

    prompt = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            prompt += f"[System]\n{content}\n"
        elif role == 'user':
            prompt += f"[User]\n{content}\n"
        elif role == 'assistant':
            prompt += f"[Assistant]\n{content}\n"
    prompt += "[Assistant]\n"
    return prompt

def extract_values(doc, query):
    text = doc.page_content
    adr_pattern = r"ADR:\s*(.*?)\s*(?=Drug:|Disease:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)"
    drug_pattern = r"Drug:\s*(.*?)\s*(?=ADR:|Disease:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)"
    disease_pattern = r"Disease:\s*(.*?)\s*(?=ADR:|Drug:|Symptom:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)"
    symptom_pattern = r"Symptom:\s*(.*?)\s*(?=ADR:|Drug:|Disease:|Other Drugs:|AnnotatorNotes:|File Path:|metadata=|$)"
    other_drugs_pattern = r"Other Drugs:\s*(.*?)\s*(?=ADR:|Drug:|Disease:|Symptom:|File Path:|metadata=|$)"

    adr = regexFunction(adr_pattern, text)
    drug = regexFunction(drug_pattern, text)
    disease = regexFunction(disease_pattern, text)
    symptom = regexFunction(symptom_pattern, text)
    other_drugs_pattern = regexFunction(other_drugs_pattern, text)

    if disease or symptom:
        return {"adr": adr, "drug": drug, "disease": disease, "symptom": symptom, "other_drugs_pattern": other_drugs_pattern}
    return {}

def regexFunction(pattern, text):
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
        merged['Drug'].update(entry.get('other_drugs_pattern', []))
        merged['Symptoms/Diseases'].update(entry.get('symptom', []))
        merged['Symptoms/Diseases'].update(entry.get('disease', []))

    final_result = {
        key: sorted(list(values)) for key, values in merged.items()
    }

    json_output = json.dumps(final_result, indent=4)
    output = os.path.join(base_dir, 'hugging_face_suggestion.json')

    with open(output, "w") as f:
        f.write(json_output)
    print('Suggested Drugs,Sytmptoms/Diseases,ADEs stored in file: hugging_face_suggestion.json')
    return json_output

def main():
    print('\nAsk your medical query: ')
    query = input()
    word_count = len(query.split())

    if(word_count < 3):
        print("Please provide more details")
    else:
        attempt = 1
        while attempt <= 3:
            print("\nAttempt: ", attempt)
            response = product_suggestion_api({"query": query})
            if response:
                data_prep_in_json(response)
                break
            else:
                print("No Response")
                global similarity_score_limit
                similarity_score_limit += 0.2
                print("Increased similarity score limit to: ", similarity_score_limit)
            attempt += 1

        if not response:
            print("Write your query in more better way")


if __name__ == "__main__":
    main()