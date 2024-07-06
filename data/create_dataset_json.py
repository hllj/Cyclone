import argparse
import os
import random
from datasets import load_dataset

from PIL import Image, ImageFile
import json

import PIL

ImageFile.LOAD_TRUNCATED_IMAGES = True

PIL.Image.MAX_IMAGE_PIXELS = 933120000

SFT_INSTRUCTION = "Cuộc trò chuyện giữa một người dùng tò mò và trợ lý trí tuệ nhân tạo. Trợ lý đưa ra câu trả lời hữu ích, chi tiết và lịch sự cho các câu hỏi của người dùng."

def create_vi_llava_dataset(local_dir):
    def process_conv_llava(conversation):
        llava_conversation = []
        for idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                if idx == 0 and "<image>" not in content:
                    content = f"<image>\n{content}" if random.random() > 0.5 else f"{content}\n<image>"
                llava_conversation.append({
                    "from": "human",
                    "value": content
                })
            elif role == "assistant":
                llava_conversation.append({
                    "from": "gpt",
                    "value": content
                })
        return llava_conversation

    def preprocess_function(local_dir, examples, path_prefix):
        id = ['llava_' + id for id in examples["id"]]
        image = [[os.path.join(local_dir, path_prefix, file_name)] for file_name in examples["file_name"]]
        for image_path in image:
            img = Image.open(image_path[0])
            img.convert("RGB").save(image_path[0])
        system = [SFT_INSTRUCTION] * len(examples["conversation"])
        return {
            "id": id,
            "images": image,
            "system": system,
            "conversations": examples["conversation"]
        }

    dataset_list = [
        "vi_llava_conversation",
        "vi_llava_complex_reasoning",
        "vi_llava_detail_description"
    ]

    split = ["train", "validation"]

    for s in split:
        total_dataset = []
        for dataset_name in dataset_list:
            print(f"Processing {dataset_name} {s}")
            dataset = load_dataset("Vi-VLM/Vista", name=dataset_name, split=s)
            if s == "train":
                path_prefix = "coco/train2017"
            elif s == "validation":
                path_prefix = "coco/val2017"
            dataset = dataset.map(lambda batch: preprocess_function(local_dir, batch, path_prefix), batched=True, remove_columns=dataset.column_names)
            dataset = dataset.filter(lambda example: (len(example["conversations"]) % 2) == 0)
            dataset = dataset.to_list()
            total_dataset.extend(dataset)
        with open(f"data/vi_llava_{s}.json", "w") as f:
            json.dump(total_dataset, f, ensure_ascii=False, indent=4)

def create_vi_sharegpt4v_dataset():
    def process_conv(conversation):
        sharegpt4v_conversation = []
        for idx, turn in enumerate(conversation):
            role = turn["from"]
            content = turn["value"]
            content = content.replace("<ảnh>", "")
            if idx == 0 and "<image>" not in content:
                content = f"<image>\n{content}" if random.random() > 0.5 else f"{content}\n<image>"
            sharegpt4v_conversation.append({
                "from": role,
                "value": content
            })
        return sharegpt4v_conversation


    def preprocess_function(examples):
        id = ['sharegpt4v_' + id for id in examples["id"]]
        image = examples["image"]
        conversations = examples["vi_conversations"]
        conversations = [process_conv(conversation) for conversation in conversations]
        return {
            "id": id,
            "image": image,
            "conversations": conversations
        }
    dataset = load_dataset("Vi-VLM/Vista", name="vi_sharegpt4v", split="train")
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.to_list()
    with open("data/vi_sharegpt4v.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def create_vi_wit_dataset(local_dir):

    def process_conv(conversation):
        llava_conversation = []
        for idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                if idx == 0 and "<image>" not in content:
                    content = f"<image>\n{content}" if random.random() > 0.5 else f"{content}\n<image>"
                llava_conversation.append({
                    "from": "human",
                    "value": content
                })
            elif role == "assistant":
                llava_conversation.append({
                    "from": "gpt",
                    "value": content
                })
        return llava_conversation

    def preprocess_function(examples):
        id = ['wit_' + id for id in examples["id"]]
        image = [f"wit/images/{id}.jpg" for id in examples["id"]]
        conversations = [process_conv(conversation) for conversation in examples["conversation"]]
        return {
            "id": id,
            "image": image,
            "conversations": conversations
        }

    def image_exists(example, local_dir):
        print("Checking image: ", example['image'])
        try:
            if os.path.exists(os.path.join(local_dir, example['image'])):
                image = Image.open(os.path.join(local_dir, example['image']))
                image.verify()
                image.close()
                print(f"Image {os.path.join(local_dir, example['image'])} exists")
                return True
            else:
                raise Exception("Image not found")
        except Exception as e:
            print(f"Error: {e} {os.path.join(local_dir, example['image'])}")
            return False

    dataset = load_dataset("Vi-VLM/Vista", name="vi_wit", split="train")
    dataset = dataset.map(lambda batch: preprocess_function(batch), batched=True, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: image_exists(example, local_dir))
    dataset = dataset.filter(lambda example: len(example["conversations"]) == 2)
    dataset = dataset.to_list()
    print("Number of examples: ", len(dataset))
    with open("data/vi_wit.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def merge_pretrain_dataset():
    with open("data/vi_sharegpt4v.json", "r") as f:
        sharegpt4v = json.load(f)
    with open("data/vi_wit.json", "r") as f:
        wit = json.load(f)

    dataset = sharegpt4v + wit
    random.shuffle(dataset)

    print("Number of examples: ", len(dataset))
    
    with open("data/vi_pretrain.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', type=str, default="/mnt/disks/dev/data/images", help='Local directory to save the dataset')
    parser.add_argument("--stage", choices=["all", "pretrain", "finetune"], type=str, default="all", help="Stage to download (all, pretrain, finetune)")
    args = parser.parse_args()
    
    if args.stage == "all" or args.stage == "finetune":
        create_vi_llava_dataset(args.local_dir)
    
    if args.stage == "all" or args.stage == "pretrain":
        create_vi_sharegpt4v_dataset()
        create_vi_wit_dataset(args.local_dir)
        merge_pretrain_dataset()