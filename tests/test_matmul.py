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
        
        # 定義維度: [Batch, Heads, M, K], [Batch, Heads, T, K]
        # M=Query Length (Decoding=1), T=Key Length, K=Head Dim
        self.B, self.nh, self.M, self.K = 1, 8, 1, 128 
        self.T = 128 
        
        # 初始化 Query 和 Key
        # 使用 float16 以符合 KIVI 的運作模式
        self.q = torch.randn((self.B, self.nh, self.M, self.K), device=self.device, dtype=torch.float16)
        self.k = torch.randn((self.B, self.nh, self.T, self.K), device=self.device, dtype=torch.float16)
        
        # 修正：Triton Kernel 強制要求 group_size 必須是 64 的倍數
        self.group_size = 64 
        self.bits = 2

    def test_triton_matmul_correctness(self):
        print("Testing Triton Matmul Correctness...")
        
        # 1. 執行量化
        # quant_and_pack_kcache 輸入 (B, nh, T, K)，壓縮 T 維度
        # scale/mn 原始輸出形狀為 (B, nh, Groups, 1, K) <--- 注意這裡有 5 維
        code, scale, mn = quant_and_pack_kcache(self.k, self.group_size, self.bits)
        
        # 2. 準備 Matmul 輸入 (關鍵修正步驟)
        # 我們需要將資料轉置成 Kernel 預期的 (B, nh, K, T_packed) 或 (B, nh, K, Groups)
        
        # 處理 Code: (B, nh, T_packed, K) -> (B, nh, K, T_packed)
        code_trans = code.transpose(2, 3).contiguous()
        
        # 處理 Scale & Mn (修正 nan 的關鍵):
        # 原始: (B, nh, Groups, 1, K)
        # 目標: (B, nh, K, Groups)
        # 步驟: squeeze(3) 去掉 1 -> (B, nh, Groups, K) -> transpose(2, 3) -> (B, nh, K, Groups)
        scale_trans = scale.squeeze(3).transpose(2, 3).contiguous()
        mn_trans = mn.squeeze(3).transpose(2, 3).contiguous()

        # 3. 執行 KIVI Triton Matmul
        output_kivi = triton_bmm_fA_qB_outer(
            self.group_size, 
            self.q, 
            code_trans, 
            scale_trans, 
            mn_trans, 
            self.bits
        )
        
        # 4. 執行標準 PyTorch Matmul (Ground Truth)
        # 使用原始的 code/scale/mn 進行解量化，確保解回來的資料形狀是 (B, nh, T, K)
        k_dequant = unpack_and_dequant_kcache(code, scale, mn, self.group_size, self.bits)
        
        # 標準運算: Q(..., 1, K) @ K_dequant.T(..., K, T)
        output_ref = torch.matmul(self.q, k_dequant.transpose(2, 3))
        
        # 5. 比較結果
        self.assertEqual(output_kivi.shape, output_ref.shape, "輸出形狀不匹配")

        # 計算誤差 (使用 float32 計算平均誤差避免溢位)
        diff = torch.abs(output_kivi - output_ref).float()
        mean_error = torch.mean(diff).item()
        print(f"  > MatMul Mean Error: {mean_error:.4f}")
        
        # 驗證誤差是否在合理範圍 (2-bit 量化誤差通常在 0.1~0.5 之間，視分佈而定)
        self.assertFalse(torch.isnan(torch.tensor(mean_error)), "Error is NaN! Kernel Memory Access Issue.")
        self.assertTrue(mean_error < 0.5, f"MatMul 誤差過大: {mean_error}")

if __name__ == '__main__':
    unittest.main()