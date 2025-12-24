
import torch

def power_compression(x_complex, gamma=0.3):
    """
    Applies power law compression to the magnitude of the complex spectrogram.
    
    Args:
        x_complex (torch.Tensor): Complex spectrogram. 
        gamma (float): Compression factor.
        
    Returns:
        torch.Tensor: Compressed magnitude and phase (concatenated) or complex representation?
                      Usually, for CRN, we feed magnitude and phase or Real/Imag parts compressed.
                      
                      Let's stick to standard practice for this paper type:
                      x_mag = |x|
                      x_phase = angle(x)
                      x_mag_compressed = x_mag ** gamma
                      
                      The model might input (x_mag_compressed, x_phase) or (x_mag_compressed * cos(phase), x_mag_compressed * sin(phase)).
                      Often called "compressed spectral magnitude" but keeping phase info.
                      
                      Let's return (mag, phase) for flexibility or the reconstructed complex with compressed mag.
                      
                      If we strictly follow "Ultra Low Complexity...", they often use:
                      input: concatenated [mag^gamma, cos(phase), sin(phase)] or similar.
                      
                      Let's implement the basic transform:
                      mag = abs(C)
                      phase = angle(C)
                      mag_compressed = mag^gamma
                      C_compressed = mag_compressed * exp(j*phase)
    """
    mag = torch.abs(x_complex)
    phase = torch.angle(x_complex)
    mag_compressed = mag.pow(gamma)
    return mag_compressed, phase

def power_decompression(mag_compressed, phase, gamma=0.3):
    """
    Reverses power law compression.
    
    Args:
        mag_compressed (torch.Tensor): Compressed magnitude.
        phase (torch.Tensor): Phase.
        gamma (float): Compression factor.
        
    Returns:
        torch.Tensor: Reconstructed complex spectrogram.
    """
    mag = mag_compressed.pow(1.0 / gamma)
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.complex(real, imag)
