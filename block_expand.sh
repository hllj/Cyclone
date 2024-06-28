#!/bin/bash

python scripts/llama_pro.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --output_dir models/Cyclone \
    --num_expand 8
