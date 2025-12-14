import unittest
import torch
from quant.new_pack import quant_and_pack_kcache, unpack_and_dequant_kcache
from quant.matmul import triton_bmm_fA_qB_outer

class TestMatMul(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if not torch.cuda.is_available():
            self.skipTest("需要 CUDA 才能測試 Triton kernel")
        self.device = 'cuda'
        
        # 定義維度: [Batch, Heads, M, K], [Batch, Heads, K, N]
        # 模擬 Query 和 Key 的矩陣乘法
        self.B, self.nh, self.M, self.K = 1, 8, 1, 128 # Query 通常長度為 1 (decoding 階段)
        self.T = 128 # Key 的長度 (SeqLen)
        
        self.q = torch.randn((self.B, self.nh, self.M, self.K), device=self.device, dtype=torch.float16)
        self.k = torch.randn((self.B, self.nh, self.T, self.K), device=self.device, dtype=torch.float16)
        
        self.group_size = 32
        self.bits = 2

    def test_triton_matmul_correctness(self):
        # 1. 準備量化後的 Key (當作矩陣 B)
        # KIVI 的 matmul 預期 Key 的形狀要做 transpose
        k_trans = self.k.transpose(2, 3).contiguous() # [B, nh, K, T]
        code, scale, mn = quant_and_pack_kcache(k_trans, self.group_size, self.bits)
        
        # 2. 執行 KIVI 的 Triton Matmul
        # 注意：這裡輸入的 code, scale, mn 需要符合 triton_bmm_fA_qB_outer 的預期維度
        # 根據原始碼，需要一些 view/reshape 操作，這裡簡化示意，實作時需參考 quant/test.py
        output_kivi = triton_bmm_fA_qB_outer(self.group_size, self.q, code, scale, mn, self.bits)
        
        # 3. 執行標準 PyTorch Matmul (作為標準答案)
        # 先解壓縮 Key
        k_dequant = unpack_and_dequant_kcache(code, scale, mn, self.group_size, self.bits)
        output_ref = torch.matmul(self.q, k_dequant)
        
        # 4. 比較兩者差異
        diff = torch.abs(output_kivi - output_ref)
        mean_error = torch.mean(diff).item()
        print(f"MatMul Mean Error: {mean_error:.4f}")
        
        # 容許一點誤差，因為 kernel 實作可能有些微精度差異
        self.assertTrue(mean_error < 0.1, "MatMul 結果與標準運算差異過大")

if __name__ == '__main__':
    unittest.main()