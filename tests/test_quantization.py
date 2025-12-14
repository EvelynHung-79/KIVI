import unittest
import torch
# 假設專案結構允許這樣 import，或是您需要設定 PYTHONPATH
from quant.new_pack import quant_and_pack_kcache, unpack_and_dequant_kcache, quant_and_pack_vcache, unpack_and_dequant_vcache

class TestQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 模擬常見的形狀: [Batch, Heads, SeqLen, HeadDim]
        self.shape = (1, 32, 128, 128)
        self.dummy_tensor = torch.randn(self.shape, device=self.device, dtype=torch.float16)
        self.group_size = 32
        self.bits = 2

    def test_k_cache_quant_dequant(self):
        print("Testing Key Cache Quantization...")
        # 1. 執行量化
        code, scale, mn = quant_and_pack_kcache(self.dummy_tensor, self.group_size, self.bits)
        
        # 2. 執行解量化
        decoded = unpack_and_dequant_kcache(code, scale, mn, self.group_size, self.bits)
        
        # 3. 驗證形狀
        self.assertEqual(decoded.shape, self.shape, "解壓縮後的形狀應該與原始輸入相同")
        
        # 4. 驗證誤差 (相對誤差不應過大，這裡設個寬鬆標準因為是 2-bit)
        error = torch.mean(torch.abs(decoded - self.dummy_tensor)).item()
        print(f"  > Mean Error: {error:.4f}")
        self.assertTrue(error < 0.5, "量化誤差過大")

    def test_v_cache_quant_dequant(self):
        print("Testing Value Cache Quantization...")
        code, scale, mn = quant_and_pack_vcache(self.dummy_tensor, self.group_size, self.bits)
        decoded = unpack_and_dequant_vcache(code, scale, mn, self.group_size, self.bits)
        
        self.assertEqual(decoded.shape, self.shape)
        error = torch.mean(torch.abs(decoded - self.dummy_tensor)).item()
        print(f"  > Mean Error: {error:.4f}")
        self.assertTrue(error < 0.5)

if __name__ == '__main__':
    unittest.main()