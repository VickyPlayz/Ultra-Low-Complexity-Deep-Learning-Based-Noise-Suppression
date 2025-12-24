
import os
import torch
import soundfile as sf
import glob
import numpy as np
from tqdm import tqdm
import argparse
from .features.feature_extraction import FeatureExtractor
from .models.crn import CRN
from .models.cnn import CNN
from .utils.metrics import calculate_metrics

def evaluate(data_dir, model_checkpoint, device='cpu'):
    # Files
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy", "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(data_dir, "clean", "*.wav")))
    
    print(f"Found {len(noisy_files)} test files.")
    
    # Load Models
    feature_extractor = FeatureExtractor().to(device)
    crn = CRN().to(device)
    cnn = CNN().to(device)
    
    checkpoint = torch.load(model_checkpoint, map_location=device)
    crn.load_state_dict(checkpoint['crn'])
    cnn.load_state_dict(checkpoint['cnn'])
    
    crn.eval()
    cnn.eval()
    
    pesq_scores = []
    stoi_scores = []
    sisnr_scores = []
    
    with torch.no_grad():
        for n_file, c_file in tqdm(zip(noisy_files, clean_files), total=100): # Limit to 100 for speed if needed, or len(noisy_files)
            # Load Audio
            noisy, sr = sf.read(n_file)
            clean, sr_c = sf.read(c_file)
            
            # Truncate/Pad to match if needed, but usually test set is aligned
            min_len = min(len(noisy), len(clean))
            noisy = noisy[:min_len]
            clean = clean[:min_len]
            
            noisy_tensor = torch.from_numpy(noisy).float().to(device).unsqueeze(0)
            
            # Inference
            noisy_mag, noisy_phase = feature_extractor(noisy_tensor)
            mag_mask = crn(noisy_mag)
            enhanced_mag = noisy_mag * mag_mask
            
            coarse_real = enhanced_mag * torch.cos(noisy_phase)
            coarse_imag = enhanced_mag * torch.sin(noisy_phase)
            cnn_input = torch.cat([coarse_real, coarse_imag], dim=1)
            
            complex_mask = cnn(cnn_input)
            
            m_r = complex_mask[:, 0:1]
            m_i = complex_mask[:, 1:2]
            i_r = coarse_real
            i_i = coarse_imag
            
            final_real = i_r * m_r - i_i * m_i
            final_imag = i_r * m_i + i_i * m_r
            
            final_mag_compressed = torch.sqrt(final_real**2 + final_imag**2 + 1e-8)
            final_phase = torch.atan2(final_imag, final_real)
            
            enhanced = feature_extractor.inverse(final_mag_compressed, final_phase)
            enhanced = enhanced.squeeze().cpu().numpy()
            
            # Match lengths
            min_len_out = min(len(clean), len(enhanced))
            clean = clean[:min_len_out]
            enhanced = enhanced[:min_len_out]
            
            # Metrics
            scores = calculate_metrics(clean, enhanced, sr)
            # print(f"Scores keys: {scores.keys()}") 
            
            if scores.get('pesq', 0.0) != 0.0:
                pesq_scores.append(scores['pesq'])
            if scores.get('stoi', 0.0) != 0.0:
                stoi_scores.append(scores['stoi'])
            
            # SI-SNR
            sisnr = scores.get('si_snr', 0.0)
            if sisnr != 0.0:
                sisnr_scores.append(sisnr)
                
            # Stop after 50 files for quick check if requested? 
            # User said "metrics evaluation", let's do full or subset.
            # VoiceBank test is ~800 files.
            
    print(f"Average PESQ: {np.mean(pesq_scores):.4f}")
    print(f"Average STOI: {np.mean(stoi_scores):.4f}")
    if sisnr_scores:
        print(f"Average SI-SNR: {np.mean(sisnr_scores):.4f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/voicebank/test", help="Path to test split")
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    evaluate(args.test_dir, args.model, args.device)
