# Llama2

## Log in to HF

* huggingface-cli login

## Download repositories

apt-get install vim
git clone https://github.com/facebookresearch/llama.git
git clone https://github.com/facebookresearch/llama-recipes/
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs

## Download HF model

* Make sure you have git-lfs installed (https://git-lfs.com)
* git lfs install
* git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## install requirements 

* pip install -r /llm/llama-recipes/requirements.txt

## Code changes to *llama-recipes*

* Create custom dataset
  * https://github.com/facebookresearch/llama-recipes/blob/main/docs/Dataset.md
  * https://github.com/tatsu-lab/stanford_alpaca
* Add `data` folder with json data file
* Change values in:
  * `llama-recipes/configs/datasets.py`
  * `llama-recipes/configs/training.py`
  * Add `llama-recipes/ft_datasets/gat_dataset.py` as copy of `llama-recipes/ft_datasets/alpaca_dataset.py`
  * `llama-recipes/ft_datasets/__init__.py`
  * `llama-recipes/utils/dataset_utils.py`

## Finetuning

General info:
* https://github.com/facebookresearch/llama-recipes/tree/main#single-gpu
* https://github.com/facebookresearch/llama-recipes/blob/main/docs/single_gpu.md
* Run from models dir:
  ```
  python3 /llm/llama-recipes/llama_finetuning.py \
  --use_peft --peft_method llama_adapter \
  --quantization \
  --dataset gat_dataset \
  --model_name Llama-2-7b-chat-hf \
  --output_dir Llama-finetuned
  ```
  
### Evaluate

* Run `inference.py`