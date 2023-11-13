import json
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_activity_labels(data_list):
    activity_label_counter = Counter()
    for data in data_list:
        activity_label_counter.update([data.get('activity_label')])
    return activity_label_counter

def count_word_frequencies(data_list):
    words_dict = Counter()
    for data in data_list:
        ctx_a = data.get('ctx_a', '')
        sentences = ctx_a.split('. ')
        for sentence in sentences:
            num_words = len(sentence.split(' '))
            words_dict.update([num_words])
    return words_dict

def create_subset(data_list, selected_labels):
    mini_train_data = [data for data in data_list if data.get('activity_label') in selected_labels]
    for i, item in enumerate(mini_train_data):
        item['ind'] = i
    return mini_train_data

def save_data(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

# Example usage
if __name__ == '__main__':
    file_path = 'data/hellaswag/hellaswag_train.jsonl'
    output_file = 'data/mixed_train.json'
    selected_labels = ['Making a sandwich', 'Disc dog', 'Surfing', 'Scuba diving', 'Fixing bicycle']

    data_list = load_data(file_path)
    activity_labels = count_activity_labels(data_list)
    word_freqs = count_word_frequencies(data_list)
    mini_train_data = create_subset(data_list, selected_labels)
    save_data(mini_train_data, output_file)

    # Print activity labels and word frequencies
    print("Activity Label Frequencies:", activity_labels)
    print("Word Frequencies:", word_freqs)
