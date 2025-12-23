import unittest
import torch
from transformers import LlamaConfig
import sys
import os
from unittest.mock import MagicMock

# 為了確保能 import 到專案根目錄的模組，將路徑加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==========================================
# Mock flash_attn 以繞過環境錯誤
# ==========================================
mock_flash = MagicMock()
mock_flash.__spec__ = MagicMock()
mock_flash.__path__ = []
sys.modules["flash_attn"] = mock_flash
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
sys.modules["flash_attn.bert_padding"] = MagicMock()
sys.modules["flash_attn_2_cuda"] = MagicMock()

try:
    from models.llama_kivi import LlamaAttention_KIVI
except ImportError:
    from models.llama_kivi import LlamaAttention_KIVI

class TestLlamaAttentionKIVI(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            k_bits=2,
            v_bits=2,
            group_size=32,
            residual_length=32, 
            use_flash=True,  
            max_position_embeddings=2048
        )
        
        self.attn = LlamaAttention_KIVI(self.config).to(self.device)
        self.attn.eval()
        self.batch_size = 1

    @unittest.skipIf(not torch.cuda.is_available(), "KIVI 需要 CUDA 環境才能執行測試")
    def test_prefill_output_structure(self):
        print("\n[Test] Running Prefill Phase Test...")
        seq_len = 10
        hidden_states = torch.randn(
            self.batch_size, seq_len, self.config.hidden_size, device=self.device
        )
        # [修正] 手動建立 position_ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        output, _, past_key_value = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids, # 傳入 position_ids
            use_cache=True
        )
        
        expected_shape = (self.batch_size, seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape, f"Output shape mismatch. Got {output.shape}, expected {expected_shape}")
        
        self.assertIsNotNone(past_key_value, "past_key_value should not be None after prefill")
        self.assertEqual(len(past_key_value), 9, "past_key_value tuple length should be 9")
        
        kv_seq_len = past_key_value[-1]
        self.assertEqual(kv_seq_len, seq_len, f"KV sequence length should be {seq_len}")

    @unittest.skipIf(not torch.cuda.is_available(), "KIVI 需要 CUDA 環境才能執行測試")
    def test_decoding_step(self):
        print("\n[Test] Running Decoding Phase Test...")
        # Step 1: Prefill
        prefill_len = 10
        hidden_states_prefill = torch.randn(
            self.batch_size, prefill_len, self.config.hidden_size, device=self.device
        )
        # [修正] Prefill 的 position_ids 為 0 ~ 9
        position_ids_prefill = torch.arange(prefill_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        _, _, past_key_value = self.attn(
            hidden_states=hidden_states_prefill, 
            position_ids=position_ids_prefill, # 傳入
            use_cache=True
        )
        
        # Step 2: Decoding
        hidden_states_decode = torch.randn(
            self.batch_size, 1, self.config.hidden_size, device=self.device
        )
        # [修正] Decoding 的 position_id 應為 [10] (接續 prefill)
        position_ids_decode = torch.tensor([[prefill_len]], dtype=torch.long, device=self.device)
        
        output, _, new_past_key_value = self.attn(
            hidden_states=hidden_states_decode,
            position_ids=position_ids_decode, # 傳入
            past_key_value=past_key_value,
            use_cache=True
        )
        
        self.assertEqual(output.shape, (self.batch_size, 1, self.config.hidden_size))
        
        old_len = past_key_value[-1]
        new_len = new_past_key_value[-1]
        self.assertEqual(new_len, old_len + 1, "KV Cache length should increase by 1")

    @unittest.skipIf(not torch.cuda.is_available(), "KIVI 需要 CUDA 環境才能執行測試")
    def test_residual_buffer_boundary(self):
        print("\n[Test] Running Residual Buffer Boundary Test...")
        target_len = self.config.residual_length 
        
        hidden_states = torch.randn(
            self.batch_size, target_len, self.config.hidden_size, device=self.device
        )
        # [修正] 手動建立 position_ids
        position_ids = torch.arange(target_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        _, _, past_key_value = self.attn(
            hidden_states=hidden_states, 
            position_ids=position_ids, # 傳入
            use_cache=True
        )
        
        key_quant = past_key_value[0] 
        key_full = past_key_value[1] 
        
        self.assertIsNotNone(key_quant, "Quantized Key Cache should created when residual buffer is full")
        self.assertIsNone(key_full, "Full Precision Key Cache should be None (cleared) after quantization trigger")
        
        print("  > Verified: Full cache cleared and Quantized cache created.")

    @unittest.skipIf(not torch.cuda.is_available(), "KIVI 需要 CUDA 環境才能執行測試")
    def test_magic_tuple_structure(self):
        print("\n[Test] Running Magic Tuple Structure Test...")
        seq_len = 5
        hidden_states = torch.randn(
            self.batch_size, seq_len, self.config.hidden_size, device=self.device
        )
        # [修正] 手動建立 position_ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

        _, _, pkv = self.attn(
            hidden_states=hidden_states, 
            position_ids=position_ids, # 傳入
            use_cache=True
        )
        
        self.assertIsNotNone(pkv[1], "idx 1 (key_states_full) should not be None for short seq")
        self.assertIsNotNone(pkv[5], "idx 5 (value_states_full) should not be None for short seq")
        
        self.assertIsNone(pkv[0], "idx 0 (key_states_quant) should be None for short seq")
        self.assertIsNone(pkv[4], "idx 4 (value_states_quant) should be None for short seq")
        
        self.assertIsInstance(pkv[8], int, "idx 8 should be an integer (kv_seq_len)")
        self.assertEqual(pkv[8], 5)

if __name__ == '__main__':
    unittest.main()