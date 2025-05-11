import os
import csv
import re
import string


def normalize_drug_name(name: str) -> str:
    """Normalize drug names by removing punctuation, plural forms, and spacing."""
    name = name.lower()
    name = name.strip(string.punctuation + "’'")
    name = re.sub(r"['’]s$", "", name)  # Remove possessive
    name = re.sub(r"s$", "", name)      # Remove plural
    name = re.sub(r"(\d+)\s*mg", r"\1mg", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def parse_annotation_file(filepath: str, base_drug: str) -> dict:
    """Parse an .ann annotation file and extract relevant entities."""
    sample_data = {
        "drug": base_drug,
        "adr": set(),
        "disease": set(),
        "symptom": set(),
        "other_drugs": set()
    }

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if not re.match(r'^T\d+\t', line):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            meta = parts[1].split()
            label = meta[0]
            text = parts[2].lower()

            if label == "Drug":
                normalized = normalize_drug_name(text)
                if normalized != base_drug:
                    sample_data["other_drugs"].add(normalized)
            elif label == "ADR":
                sample_data["adr"].add(text)
            elif label == "Disease":
                sample_data["disease"].add(text)
            elif label == "Symptom":
                sample_data["symptom"].add(text)

    return sample_data


def process_annotations(data_dir: str) -> list:
    """Process all .ann files in the given directory and return structured data."""
    data = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".ann"):
            continue

        filepath = os.path.join(data_dir, filename)
        match = re.search(r'Original/([^./]+)\.', filepath)
        if not match:
            continue

        base_drug = match.group(1).lower()
        sample_data = parse_annotation_file(filepath, base_drug)
        data.append(sample_data)

    return data


def write_to_csv(data: list, output_file: str):
    """Write structured data to a CSV file."""
    fieldnames = ['Drug', 'Disease', 'Symptom', 'ADR', 'Other Drugs']

    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for entry in data:
            row = [
                entry['drug'],
                ", ".join(sorted(entry['disease'])),
                ", ".join(sorted(entry['symptom'])),
                ", ".join(sorted(entry['adr'])),
                ", ".join(sorted(entry['other_drugs']))
            ]
            writer.writerow(row)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_data_dir = os.path.join(base_dir, "CADEC.v1/Original")
    output_csv_path = os.path.join(base_dir, "data_set.csv")

    all_data = process_annotations(original_data_dir)
    write_to_csv(all_data, output_csv_path)

    print(f"CSV written to {output_csv_path}")


if __name__ == "__main__":
    main()