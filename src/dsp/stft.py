
import torch

def stft(x, n_fft=512, hop_length=256, win_length=512):
    """
    Computes the Short-Time Fourier Transform (STFT).
    
    Args:
        x (torch.Tensor): Input signal of shape (batch, time) or (time).
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        win_length (int): Window length.
        
    Returns:
        torch.Tensor: Complex spectrogram of shape (batch, n_fft//2 + 1, frames, 2)
                      where last dim is (real, imag).
    """
    window = torch.hann_window(win_length, device=x.device)
    return torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)

def istft(x_complex, n_fft=512, hop_length=256, win_length=512):
    """
    Computes the Inverse Short-Time Fourier Transform (iSTFT).
    
    Args:
        x_complex (torch.Tensor): Complex spectrogram of shape (batch, freq, frames).
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        win_length (int): Window length.
        
    Returns:
        torch.Tensor: Reconstructed signal of shape (batch, time).
    """
    window = torch.hann_window(win_length, device=x_complex.device)
    return torch.istft(x_complex, n_fft, hop_length, win_length, window)
