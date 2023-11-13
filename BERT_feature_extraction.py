from transformers import BertTokenizer, BertModel
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def extract_embeddings(data_list, tokenizer, model):
    node_embeddings = []
    with torch.no_grad():
        for data in data_list:
            ctx_a = data.get("ctx_a")
            encoded = tokenizer.encode_plus(ctx_a, add_special_tokens=True, max_length=80, padding='max_length', return_attention_mask=True, return_tensors='pt')
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state.mean(1).squeeze()
            normalized_output = torch.nn.functional.normalize(pooled_output, dim=0)
            node_embeddings.append(normalized_output)
    return node_embeddings
