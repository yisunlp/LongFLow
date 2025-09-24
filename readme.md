## Quick start

### requirement
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51
```

### usage
```
python example_multi_batch.py --model_name ${MODEL_NAME_OR_PATH} --method ${METHOD}

# e.g.
python example_multi_batch.py --model_name Qwen/Qwen3-8B --method ours
```
