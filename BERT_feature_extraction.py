from transformers import BertTokenizer, BertModel
import torch

def set_seed(seed):
    torch.manual_seed(seed)

def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def get_bert_embeddings(data_list, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        for data in data_list:
            ctx_a = data.get("ctx_a", "")
            encoded = tokenizer.encode_plus(ctx_a, add_special_tokens=True, max_length=80, padding='max_length', return_attention_mask=True, return_tensors='pt')
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state.mean(1).squeeze()
            embeddings.append(torch.nn.functional.normalize(pooled_output, dim=0))
    return embeddings

# Example usage
if __name__ == '__main__':
    set_seed(42)
    tokenizer, model = load_bert_model()
    # Assume data_list is loaded from the previous script
    embeddings = get_bert_embeddings(data_list, tokenizer, model)
