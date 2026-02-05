# pip install -U huggingface_hub pandas hf-xet

import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "vislupus/bulgarian-dictionary-raw-data"
out_dir = "bg_dict_csv"
os.makedirs(out_dir, exist_ok=True)

files = list_repo_files(repo_id=repo_id, repo_type="dataset")
csvs = [f for f in files if f.lower().endswith(".csv")]
print("Found CSVs:", csvs)

for fname in csvs:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname, local_dir=out_dir)
    print("Downloaded:", local_path)
