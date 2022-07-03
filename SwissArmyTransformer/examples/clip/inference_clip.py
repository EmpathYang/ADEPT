import os
import argparse
from SwissArmyTransformer.model.official.clip_model import CLIP, ImageEncoder
from SwissArmyTransformer import get_args
py_parser = argparse.ArgumentParser(add_help=False)
py_parser.add_argument('--old_checkpoint', action="store_true")
py_parser.add_argument('--md_type', type=str)
py_parser = CLIP.add_model_specific_args(py_parser)
py_parser = ImageEncoder.add_model_specific_args(py_parser)
known, args_list = py_parser.parse_known_args()
args = get_args(args_list)
args = argparse.Namespace(**vars(args), **vars(known))
model_type = args.md_type

import os
import torch
model, args = CLIP.from_pretrained(args, model_type)
# from SwissArmyTransformer.training.deepspeed_training import load_checkpoint

# model = CLIP(args)
# load_checkpoint(model, args)
# model = model.cuda()

from transformers import CLIPProcessor
processor = CLIPProcessor.from_pretrained(os.path.join('', 'openai/clip-vit-base-patch32'))

model.eval()
with torch.no_grad():
    from PIL import Image
    import requests
    # import matplotlib.pyplot as plt

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # plt.imshow(image)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
    )
    expanded_mask = inputs['attention_mask'][:, None, None, :].expand(2, 1, 7, 7).to(torch.float)

    image_position_ids = torch.cat([torch.arange(50)[None,]])
    text_position_ids = torch.cat([torch.arange(7)[None,], torch.arange(7)[None,]])
    encoded_input = {'text_attention_mask':expanded_mask, 'image_input_ids':torch.zeros(1, 1).long(), 'image_position_ids':image_position_ids, 'image':inputs['pixel_values'], 'text_input_ids':inputs['input_ids'], 'text_position_ids':text_position_ids}
    model = model.cuda()
    encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
    image_embeds, text_embeds, logits_per_text, logits_per_image = model(offline=True, **encoded_input)
    logits_per_text = logits_per_text.cpu()
    print(logits_per_text)

# breakpoint()
