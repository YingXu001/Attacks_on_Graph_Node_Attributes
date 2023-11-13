import json

# Paths to the JSONL files
files = [
    'C:\\Users\\fiona\\Master Thesis\\AttackGCN\\data\\hellaswag\\hellaswag_train.jsonl',
    'C:\\Users\\fiona\\Master Thesis\\AttackGCN\\data\\hellaswag\\hellaswag_test.jsonl',
    'C:\\Users\\fiona\\Master Thesis\\AttackGCN\\data\\hellaswag\\hellaswag_val.jsonl'
]

# List to store all JSON objects
all_data = []

# Reading each JSONL file and adding its content to the list
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            # Convert each line into a JSON object and append to the list
            all_data.append(json.loads(line))

# Writing the combined list to a new JSON file
output_file = 'C:\\Users\\fiona\\Master Thesis\\Attack_Graph\\data\\combined_hellaswag.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("Combination Complete. The combined file is saved as:", output_file)
