# quant/cache_manager.py
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from quant.new_pack import triton_quantize_and_pack_along_last_dim

@dataclass
class KiviCacheState:
    """
    Encapsulates the KIVI cache state to replace the 8-element magic tuple.
    """
    key_quant: Optional[torch.Tensor] = None
    key_full: Optional[torch.Tensor] = None
    key_scale: Optional[torch.Tensor] = None
    key_zero_point: Optional[torch.Tensor] = None  # Renamed from mn
    value_quant: Optional[torch.Tensor] = None
    value_full: Optional[torch.Tensor] = None
    value_scale: Optional[torch.Tensor] = None
    value_zero_point: Optional[torch.Tensor] = None # Renamed from mn
    current_length: int = 0

    def to_tuple(self) -> Tuple:
        """Maintains backward compatibility for HuggingFace's cache interface."""
        return (
            self.key_quant, self.key_full, self.key_scale, self.key_zero_point,
            self.value_quant, self.value_full, self.value_scale, self.value_zero_point,
            self.current_length
        )

    @classmethod
    def from_tuple(cls, t: Tuple):
        """Helper to restore state from tuple."""
        if t is None:
            return cls()
        # Ensure we handle cases where tuple might not have all elements if legacy
        return cls(
            key_quant=t[0], key_full=t[1], key_scale=t[2], key_zero_point=t[3],
            value_quant=t[4], value_full=t[5], value_scale=t[6], value_zero_point=t[7],
            current_length=t[8]
        )

class KiviCacheManager:
    """
    Handles the lifecycle of KIVI Cache: quantization, buffering, and updates.
    """
    def __init__(self, k_bits, v_bits, group_size, residual_length):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length

    def update_key_state(self, cache: KiviCacheState, new_key_states: torch.Tensor, is_prefill: bool = False):
        """Updates the Key Cache: appends new data and triggers quantization if needed."""
        if is_prefill:
            # Prefill Logic
            seq_len = new_key_states.shape[-2]
            if seq_len % self.residual_length != 0:
                if seq_len < self.residual_length:
                    cache.key_quant = None
                    cache.key_full = new_key_states
                else:
                    split_idx = -(seq_len % self.residual_length)
                    key_to_quant = new_key_states[:, :, :split_idx, :].contiguous()
                    cache.key_full = new_key_states[:, :, split_idx:, :].contiguous()
                    
                    cache.key_quant, cache.key_scale, cache.key_zero_point = triton_quantize_and_pack_along_last_dim(
                        key_to_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                    )
            else:
                cache.key_full = None
                cache.key_quant, cache.key_scale, cache.key_zero_point = triton_quantize_and_pack_along_last_dim(
                    new_key_states.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                )
        else:
            # Decoding Logic
            if cache.key_full is not None:
                key_states_full_new = torch.cat([cache.key_full, new_key_states], dim=2)
            else:
                key_states_full_new = new_key_states

            if key_states_full_new.shape[-2] == self.residual_length:
                k_quant_new, k_scale_new, k_zp_new = triton_quantize_and_pack_along_last_dim(
                    key_states_full_new.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                )
                cache.key_full = None
                
                if cache.key_quant is not None:
                    cache.key_quant = torch.cat([cache.key_quant, k_quant_new], dim=3)
                    cache.key_scale = torch.cat([cache.key_scale, k_scale_new], dim=3)
                    cache.key_zero_point = torch.cat([cache.key_zero_point, k_zp_new], dim=3)
                else:
                    cache.key_quant = k_quant_new
                    cache.key_scale = k_scale_new
                    cache.key_zero_point = k_zp_new
            else:
                cache.key_full = key_states_full_new

    def update_value_state(self, cache: KiviCacheState, new_value_states: torch.Tensor, is_prefill: bool = False):
        """Updates the Value Cache: appends new data and triggers quantization if needed."""
        if is_prefill:
            # Prefill Logic
            seq_len = new_value_states.shape[-2]
            if seq_len <= self.residual_length:
                cache.value_quant = None
                cache.value_full = new_value_states
                cache.value_scale = None
                cache.value_zero_point = None
            else:
                split_idx = -self.residual_length
                value_to_quant = new_value_states[:, :, :split_idx, :].contiguous()
                cache.value_full = new_value_states[:, :, split_idx:, :].contiguous()
                
                cache.value_quant, cache.value_scale, cache.value_zero_point = triton_quantize_and_pack_along_last_dim(
                    value_to_quant, self.group_size, self.v_bits
                )
        else:
            # Decoding Logic
            cache.value_full = torch.cat([cache.value_full, new_value_states], dim=2) if cache.value_full is not None else new_value_states
            value_full_length = cache.value_full.shape[-2]
            
            if value_full_length > self.residual_length:
                v_quant_new, v_scale_new, v_zp_new = triton_quantize_and_pack_along_last_dim(
                    cache.value_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits
                )
                cache.value_full = cache.value_full[:, :, 1:, :].contiguous()
                
                if cache.value_quant is not None:
                    cache.value_quant = torch.cat([cache.value_quant, v_quant_new], dim=2)
                    cache.value_scale = torch.cat([cache.value_scale, v_scale_new], dim=2)
                    cache.value_zero_point = torch.cat([cache.value_zero_point, v_zp_new], dim=2)
                else:
                    cache.value_quant = v_quant_new
                    cache.value_scale = v_scale_new
                    cache.value_zero_point = v_zp_new
