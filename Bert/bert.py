

# import torch
# from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained("hfll/chinese-roberta-wwm-ext")
# roberta = BertModel.from_pretrained("hfll/chinese-roberta-wwm-ext")
# inputs = tokenizer("试试")
# print(inputs)
# outputs = roberta(inputs)
# print(outputs)

import torch
from transformers import AutoModel, AutoTokenizer

def bert_encoder(texts, model, tokenizer, device):
    # print('begin')
    input_ids, token_type_ids, attention_masks = [[101] for t in texts], [[0] for t in texts], [[1] for t in texts]
    for index_text, text in enumerate(texts):
        for index_t, t in enumerate(text):
            tok_res = tokenizer(t, add_special_tokens=False)
            if tok_res['input_ids']:
                input_ids[index_text].append(tok_res['input_ids'][0])
                token_type_ids[index_text].append(tok_res['token_type_ids'][0])
                attention_masks[index_text].append(tok_res['attention_mask'][0])
    # input_ids = [input_id[:300] for input_id in input_ids]
    # #print(input_ids)
    # token_type_ids = [token_type_id[:300] for token_type_id in token_type_ids]
    # #print(token_type_ids)
    # attention_masks = [attention_mask[:300] for attention_mask in attention_masks]
    for input_id in input_ids:
        input_id.append(102)
    for token_type_id in token_type_ids:
        token_type_id.append(0)
    for attention_mask in attention_masks:
        attention_mask.append(1)
    max_len = max([len(input_id) for input_id in input_ids])
    input_ids = [input_id + ([0] * (max_len - len(input_id))) for input_id in input_ids]
    token_type_ids = [token_type_id + ([0] * (max_len - len(token_type_id))) for token_type_id in token_type_ids]
    attention_masks = [attention_mask + ([0] * (max_len - len(attention_mask))) for attention_mask in attention_masks]

    input_ids = torch.tensor(input_ids).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)
    # print(input_ids)
    # print(token_type_ids)
    # print(attention_masks)
    
    model_encode = model(input_ids, token_type_ids, attention_masks)
   
    # for index_text, text in enumerate(texts):
    #     #print(len(model_encode[0][index_text]))
    #     #print(model_encode[0][index_text])
    #     #print(torch.mean(model_encode[0][index_text], dim=0))
    
    encode = torch.mean(model_encode[0], dim=1)
        # #print(len(encoding))
        # features.append(encoding)
    #print(len(features))
    # print('end')
    return encode.detach().squeeze().tolist()

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# roberta = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

def bert_encoder2(texts):
    #print(len(texts))
    device = 'cuda'
    features = []
    input_ids, token_type_ids, attention_masks = [[101] for t in texts], [[0] for t in texts], [[1] for t in texts]
    for index_text, text in enumerate(texts):
        for index_t, t in enumerate(text):
            tok_res = tokenizer(t, add_special_tokens=False)
            #print(tok_res)
            if tok_res['input_ids']:
                input_ids[index_text].append(tok_res['input_ids'][0])
                token_type_ids[index_text].append(tok_res['token_type_ids'][0])
                attention_masks[index_text].append(tok_res['attention_mask'][0])
    input_ids = [input_id[:510] for input_id in input_ids]
    #print(input_ids)
    token_type_ids = [token_type_id[:510] for token_type_id in token_type_ids]
    #print(token_type_ids)
    attention_masks = [attention_mask[:510] for attention_mask in attention_masks]
    for input_id in input_ids:
        input_id.append(102)
    for token_type_id in token_type_ids:
        token_type_id.append(0)
    for attention_mask in attention_masks:
        attention_mask.append(1)
    max_len = max([len(input_id) for input_id in input_ids])
    input_ids = [input_id + ([0] * (max_len - len(input_id))) for input_id in input_ids]
    token_type_ids = [token_type_id + ([0] * (max_len - len(token_type_id))) for token_type_id in token_type_ids]
    attention_masks = [attention_mask + ([0] * (max_len - len(attention_mask))) for attention_mask in attention_masks]

    input_ids = torch.tensor(input_ids).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)
    model_encode = model(input_ids, token_type_ids, attention_masks)
    for index_text, text in enumerate(texts):
        #print(len(model_encode[0][index_text]))
        #print(model_encode[0][index_text])
        #print(torch.mean(model_encode[0][index_text], dim=0))
        encoding = torch.mean(model_encode[0][index_text], dim=0).tolist()
        #print(len(encoding))
        features.append(encoding)
    #print(len(features))
    return features

if __name__ == '__main__':
    dev = 'cuda'
    model_dir = 'chinese_roberta_wwm_ext_pytorch'
    model = AutoModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = model.to(dev)
    output = bert_encoder(["我是我"], model,tokenizer, dev)
    print('1',output[:10], output[-10:])
    output = bert_encoder2(["我是我"])[0]
    print('2',output[:10], output[-10:])
    # print(len(output[0]))

    