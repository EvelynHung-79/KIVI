import torch
# import kivi
from quant.new_pack import quant_and_pack_kcache

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"KIVI installed successfully!")

# 簡單測試一下量化功能有沒有崩潰
try:
    dummy = torch.randn((1, 32, 128, 128), device='cuda', dtype=torch.float16)
    quant_and_pack_kcache(dummy, 32, 2)
    print("Core Quantization Function: OK ✅")
except Exception as e:
    print(f"Quantization Failed ❌: {e}")