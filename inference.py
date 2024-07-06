import requests
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, TextStreamer

model_id = "models/paligemma-3b-mix-224-vi-llava"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

image_path = "images/example_1.jpg"
image = Image.open(image_path)
pixel_values = processor(images=[image], return_tensors="pt").to(model.device)["pixel_values"]

messages = [
    {"role": "user", "content": "Đây là nơi nào ở Việt Nam?"}
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
image_token_id = tokenizer.convert_tokens_to_ids("<image>")
image_prefix = torch.empty((1, getattr(processor, "image_seq_length")), dtype=input_ids.dtype).fill_(image_token_id)
input_ids = torch.cat((image_prefix, input_ids), dim=-1).to(model.device)

output_ids = model.generate(input_ids, pixel_values=pixel_values, streamer=streamer, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.7)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output)