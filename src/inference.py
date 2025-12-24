
import os
import torch
import soundfile as sf
import argparse
from .features.feature_extraction import FeatureExtractor
from .models.crn import CRN
from .models.cnn import CNN
from .dsp.compression import power_decompression

def enhance_file(input_path, output_path, model_checkpoint, device='cpu'):
    # Load Models
    feature_extractor = FeatureExtractor().to(device)
    crn = CRN().to(device)
    cnn = CNN().to(device)
    
    # Load Weights
    checkpoint = torch.load(model_checkpoint, map_location=device)
    crn.load_state_dict(checkpoint['crn'])
    cnn.load_state_dict(checkpoint['cnn'])
    
    crn.eval()
    cnn.eval()
    
    # Process
    with torch.no_grad():
        noisy, sr = sf.read(input_path)
        noisy_tensor = torch.from_numpy(noisy).float().to(device).unsqueeze(0) # (1, T)
        
        # 1. Features
        noisy_mag, noisy_phase = feature_extractor(noisy_tensor) 
        
        # 2. CRN
        mag_mask = crn(noisy_mag)
        enhanced_mag = noisy_mag * mag_mask
        
        # 3. Intermediate to Complex for CNN
        coarse_real = enhanced_mag * torch.cos(noisy_phase)
        coarse_imag = enhanced_mag * torch.sin(noisy_phase)
        cnn_input = torch.cat([coarse_real, coarse_imag], dim=1)
        
        # 4. CNN
        complex_mask = cnn(cnn_input)
        
        # 5. Final Enhance (Complex Mul)
        m_r = complex_mask[:, 0:1]
        m_i = complex_mask[:, 1:2]
        i_r = coarse_real
        i_i = coarse_imag
        
        final_real = i_r * m_r - i_i * m_i
        final_imag = i_r * m_i + i_i * m_r
        
        # Reconstruct
        # feature_extractor.inverse expects mag_compressed, phase OR we can use istft directly if we have complex.
        # feature_extractor.inverse does power_decompression first.
        # But we have final_real, final_imag which are compressed domain complex values?
        # Yes, cnn_input was compressed-domain complex.
        # So final_real/imag are compressed-domain.
        # We need to decompress them.
        
        # Reconstruct magnitude and phase from final_real, final_imag
        final_mag_compressed = torch.sqrt(final_real**2 + final_imag**2 + 1e-8)
        final_phase = torch.atan2(final_imag, final_real)
        
        # Decompress
        # feature_extractor.inverse takes (mag_compressed, phase) (B, 1, T, F)
        # Reshape to (B, 1, T, F) -> it is already (B, 1, T, F) from CNN output?
        # CNN output is (B, 2, T, F) or we computed final_real/imag as (B, 1, T, F).
        
        final_audio = feature_extractor.inverse(final_mag_compressed, final_phase)
        
        # Save
        sf.write(output_path, final_audio.squeeze().cpu().numpy(), sr)
        print(f"Enhanced audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input noisy file")
    parser.add_argument("--output", "-o", required=True, help="Output enhanced file")
    parser.add_argument("--model", "-m", required=True, help="Path to checkpoint")
    parser.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    enhance_file(args.input, args.output, args.model, args.device)
