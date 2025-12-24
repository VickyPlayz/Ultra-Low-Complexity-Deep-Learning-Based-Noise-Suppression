
import torch
import torch.nn as nn
import torch.nn.functional as F

def si_snr(estimate, reference, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    Args:
        estimate (torch.Tensor): Estimated signal (B, T).
        reference (torch.Tensor): Reference signal (B, T).
    Returns:
        torch.Tensor: Scalar SI-SNR value.
    """
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)
    
    reference_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    optimal_scaling = torch.sum(reference * estimate, dim=-1, keepdim=True) / reference_energy
    
    projection = optimal_scaling * reference
    noise = estimate - projection
    
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)

class LossFunction(nn.Module):
    def __init__(self, heavy_hitters_alpha=0.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, est_mag, ref_mag, est_complex, ref_complex):
        """
        Composite loss: Magnitude Loss + Complex Loss
        Args:
            est_mag: Estimated Magnitude (B, T, F)
            ref_mag: Reference Magnitude (B, T, F)
            est_complex: Estimated Complex (B, 2, T, F) or (B, T, F, 2)
            ref_complex: Reference Complex
        """
        # Magnitude Loss (L2 or L1)
        mag_loss = F.mse_loss(est_mag, ref_mag)
        
        # Complex Loss (L2 on Real/Imag)
        # Assuming shape (B, 2, T, F) -> permute to align if needed or just MSE
        complex_loss = F.mse_loss(est_complex, ref_complex)
        
        return mag_loss + complex_loss


try:
    from pesq import pesq
except ImportError:
    pesq = None
    print("Warning: PESQ not installed. PESQ scores will be 0.")

try:
    from pystoi import stoi
except ImportError:
    stoi = None
    print("Warning: STOI not installed. STOI scores will be 0.")

import numpy as np

def calculate_metrics(clean, enhanced, sample_rate=16000):
    """
    Calculate PESQ and STOI.
    Args:
        clean (torch.Tensor or np.ndarray): Clean signal (1D).
        enhanced (torch.Tensor or np.ndarray): Enhanced signal (1D).
        sample_rate (int): Sampling rate (default 16000).
        
    Returns:
        dict: {'pesq': float, 'stoi': float}
    """
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy().squeeze()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy().squeeze()
        
    # Ensure they are 1D
    if clean.ndim > 1:
        clean = clean.reshape(-1)
    if enhanced.ndim > 1:
        enhanced = enhanced.reshape(-1)
        
    # STOI
    stoi_score = 0.0
    if stoi is not None:
        try:
            stoi_score = stoi(clean, enhanced, sample_rate, extended=False)
        except Exception as e:
            print(f"STOI Error: {e}")
            pass
        
    # PESQ
    pesq_score = 0.0
    if pesq is not None:
        try:
            # mode 'wb' for 16k, 'nb' for 8k
            mode = 'wb' if sample_rate == 16000 else 'nb'
            pesq_score = pesq(sample_rate, clean, enhanced, mode)
        except Exception as e:
            pass
        

    # SI-SNR
    # Assume clean/enhanced are 1D floats. Convert to torch for si_snr function or reimplement?
    # The file already has 'si_snr' function at the top.
    # We need to make sure inputs are tensor for that function, or numpy approach.
    # The existing 'si_snr' function in this file takes (estimate, reference).
    # It expects tensors (B, T) usually. 
    
    # Let's wrap it lightly.
    try:
        clean_t = torch.from_numpy(clean).float().unsqueeze(0)
        enhanced_t = torch.from_numpy(enhanced).float().unsqueeze(0)
        sisnr_val = si_snr(enhanced_t, clean_t)
        sisnr_score = sisnr_val.item()
    except Exception as e:
        sisnr_score = 0.0
        
    return {'pesq': pesq_score, 'stoi': stoi_score, 'si_snr': sisnr_score}
