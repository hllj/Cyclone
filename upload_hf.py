from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="saves/paligemma-chat/checkpoint-10000",
    repo_id="hllj/paligemma-3b-mix-224-vi-llava-checkpoint-10000",
    repo_type="model"
)