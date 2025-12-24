
import torch
import torch.nn as nn
from ..dsp.stft import stft, istft
from ..dsp.compression import power_compression, power_decompression

class FeatureExtractor(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512, compress_gamma=0.3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compress_gamma = compress_gamma
        
        # Register window as buffer to be saved with model
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x):
        """
        Extracts features from input audio.
        
        Args:
            x (torch.Tensor): Input audio (batch, samples).
            
        Returns:
            torch.Tensor: Compressed magnitude features (batch, channels, time, freq).
            torch.Tensor: Phase (batch, channels, time, freq).
        """
        # Compute STFT
        # stft returns (batch, freq, time) complex? 
        # Our src/dsp/stft.py returns complex tensor. 
        # Checking src/dsp/stft.py: 
        # return torch.stft(..., return_complex=True) -> Shape (..., freq, frames) usually? 
        # Let's verify shape. torch.stft documentation: (..., n_fft//2 + 1, frames).
        
        x_complex = stft(x, self.n_fft, self.hop_length, self.win_length)
        
        # Power Compression
        mag_compressed, phase = power_compression(x_complex, self.compress_gamma)
        
        # Shape: (batch, freq, frames)
        # We usually want (batch, channels, time, freq) for Conv2D.
        # Let's permute.
        
        mag_compressed = mag_compressed.permute(0, 2, 1).unsqueeze(1) # (B, 1, T, F)
        phase = phase.permute(0, 2, 1).unsqueeze(1)                   # (B, 1, T, F)
        
        return mag_compressed, phase

    def inverse(self, mag_compressed, phase):
        """
        Reconstructs audio from features.
        """
        # (B, 1, T, F) -> (B, F, T)
        mag_compressed = mag_compressed.squeeze(1).permute(0, 2, 1)
        phase = phase.squeeze(1).permute(0, 2, 1)
        
        x_complex = power_decompression(mag_compressed, phase, self.compress_gamma)
        
        return istft(x_complex, self.n_fft, self.hop_length, self.win_length)
