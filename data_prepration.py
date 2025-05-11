import os
import csv
import re
import string

base_dir = os.path.dirname(os.path.abspath(__file__))
original_data_set = os.path.join(base_dir, "CADEC.v1/Original")
formatted_data = os.path.join(base_dir, "data_set.csv")

def normalize_drug_name(name):
    name = name.lower()
    name = name.strip(string.punctuation + "’'")
    name = re.sub(r"['’]s$", "", name)            
    name = re.sub(r"s$", "", name)
    name = re.sub(r"(\d+)\s*mg", r"\1mg", name)   
    name = re.sub(r"[^a-z0-9]", "", name)         
    return name

all_data_array = []
# Process each .ann file
for filename in os.listdir(original_data_set):
    if filename.endswith(".ann"):
        filepath = os.path.join(original_data_set, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            # print(filepath)
            match = re.search(r'Original/([^./]+)\.', filepath)
            if not match:
                continue
            drug = match.group(1).lower()
            sample_data = {"drug": drug, "adr": set(), "disease": set(), "symptom": set(), "other_drugs": set()}
            for line in file:
                line = line.strip()

                if re.match(r'^T\d+\t', line):
                    parts = line.split('\t')
                    tid = parts[0]
                    meta = parts[1].split()
                    label = meta[0]
                    text = parts[2].lower()

                    if label == "Drug":
                        text = normalize_drug_name(text)
                        if text == drug:
                            continue
                        sample_data["other_drugs"].add(text)
                    elif label == "ADR":
                        sample_data["adr"].add(text)
                    elif label == "Disease":
                        sample_data["disease"].add(text)
                    elif label == "Symptom":
                        sample_data["symptom"].add(text)
                # print(sample_data)

            all_data_array.append(sample_data)


with open(formatted_data, "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ['Drug', 'Disease', 'Symptom', 'ADR' ,'Other Drugs']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)

    for data in all_data_array:
        print(data)
        row = [
            data['drug'],
            ", ".join(sorted(data["disease"])),
            ", ".join(sorted(data["symptom"])),
            ", ".join(sorted(data["adr"])),
            ", ".join(sorted(data["other_drugs"]))
        ]
        writer.writerow(row)
        # break

print(f"CSV written to {formatted_data}")