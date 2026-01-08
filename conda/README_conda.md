# Conda environments

We provide three environments. Choose based on your platform.

## 1) CPU-only (simulate HRL without probing)
```bash
conda env create -f conda/environment.cpu.yml
conda activate hrl3d-cpu
pip install -e ".[hf]"   # optional for GPT-2 introspection
pip install -e ".[rl]"   # required for HRL training
```

## 2) CUDA GPU probing + HRL
```bash
conda env create -f conda/environment.cuda.yml
conda activate hrl3d-cuda
pip install -e ".[rl,hf]"
```

## 3) Ascend NPU probing (best-effort) + HRL
Ascend requires vendor-specific runtime (CANN + torch_npu). Create env first, then install vendor packages following your platform guide.
```bash
conda env create -f conda/environment.ascend.yml
conda activate hrl3d-ascend
pip install -e ".[rl,hf]"
# then install torch_npu per your environment
```
