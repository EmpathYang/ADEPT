import os
import torch
import argparse
from SwissArmyTransformer import get_args, AutoModel
# from SwissArmyTransformer.model.official.bert_model import BertModel

args = get_args()

model_type = 'bert-base-uncased'
model, args = AutoModel.from_pretrained(args, model_type)

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained(os.path.join('', model_type))
bert = BertForMaskedLM.from_pretrained(os.path.join('', model_type), output_hidden_states=True)

model.eval()
bert.eval()
with torch.no_grad():
    text = [["This is a piece of text.", "Another piece of text."]]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    hugging_output = bert(**encoded_input)[0]
    print(position_ids)
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    # Since we don't use padding_idx for Embedding layers, pad output is largely different between hugging and swiss.
    # You will find it if you calculate error for hugging_output[1] and swiss_output[1].
    # However, pad output is usually not used, it doesn't matter too much.
    print("max error:", (hugging_output[:,0] - swiss_output[:,0]).abs().max())
    print("max relative error:", ((hugging_output[:,0] - swiss_output[:,0]).abs() / torch.max(swiss_output[:,0].abs(), hugging_output[:,0].abs())).max())

# breakpoint()