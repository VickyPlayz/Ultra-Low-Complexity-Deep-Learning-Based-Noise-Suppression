
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import soundfile as sf
import random
from .features.feature_extraction import FeatureExtractor
from .models.crn import CRN
from .models.cnn import CNN
from .utils.metrics import LossFunction

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, max_len=None):
        """
        Args:
            data_dir: Path to split directory (e.g., data/voicebank/train)
        """
        self.noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy", "*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(data_dir, "clean", "*.wav")))
        self.sample_rate = sample_rate
        self.max_len = max_len # Truncate/Pad
        
        # Verify alignment
        # assert len(self.noisy_files) == len(self.clean_files)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, sr_n = sf.read(self.noisy_files[idx])
        clean, sr_c = sf.read(self.clean_files[idx])
        
        # Normalize?
        # Usually float32 is already normalized -1 to 1.
        
        # Pad/Crop
        # For training, usually fixed length chunks are used.
        # Let's say 3 seconds.
        chunk_len = 3 * self.sample_rate
        if len(noisy) < chunk_len:
            pad = chunk_len - len(noisy)
            noisy = torch.nn.functional.pad(torch.from_numpy(noisy), (0, pad))
            clean = torch.nn.functional.pad(torch.from_numpy(clean), (0, pad))
        else:
            # Random crop
            start = random.randint(0, len(noisy) - chunk_len)
            noisy = torch.from_numpy(noisy[start:start+chunk_len])
            clean = torch.from_numpy(clean[start:start+chunk_len])
            
        return noisy.float(), clean.float()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-3
    DATA_ROOT = os.path.join("data", "voicebank")
    
    # Loaders
    train_ds = AudioDataset(os.path.join(DATA_ROOT, "train"))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 for Windows robustness
    
    # Models
    feature_extractor = FeatureExtractor().to(device)
    crn = CRN().to(device)
    cnn = CNN().to(device)
    
    optimizer = torch.optim.Adam(list(crn.parameters()) + list(cnn.parameters()), lr=LR)
    criterion = LossFunction().to(device)
    
    for epoch in range(EPOCHS):
        crn.train()
        cnn.train()
        total_loss = 0
        
        for i, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # Forward
            # 1. Features
            noisy_mag, noisy_phase = feature_extractor(noisy) # (B, 1, T, F), (B, 1, T, F)
            clean_mag, clean_phase = feature_extractor(clean) # (B, 1, T, F), (B, 1, T, F)
            
            # Prepare clean complex ref
            # Clean Complex = clean_mag * exp(j * clean_phase)
            # Or just use the output of inverse? inverse takes compressed mag.
            # Let's assume loss computed on Compressed Mag and Resynthesized Complex.
            
            # 2. CRN (Stage 1)
            mag_mask = crn(noisy_mag)
            enhanced_mag = noisy_mag * mag_mask
            
            # 3. Intermediate Reconstruction
            # coarse_enhanced_complex = (enhanced_mag, noisy_phase)
            # But CNN takes (Real, Imag)
            
            # Convert (mag, phase) -> (Real, Imag)
            # Note: feature_extractor.inverse does decompression
            # We need the compressed complex representation for CNN? 
            # Or the decompressed?
            # Typically CNN operates on compressed feature domain or decompressed.
            # Let's assume it operates on the same domain as features (compressed mag).
            # So we create "Coarse Complex" from (enhanced_mag, noisy_phase)
            
            coarse_real = enhanced_mag * torch.cos(noisy_phase)
            coarse_imag = enhanced_mag * torch.sin(noisy_phase)
            cnn_input = torch.cat([coarse_real, coarse_imag], dim=1) # (B, 2, T, F)
            
            # 4. CNN (Stage 2)
            complex_mask = cnn(cnn_input) # (B, 2, T, F)
            
            # 5. Final Enhance
            # Additive or Multiplicative mask?
            # "cIRM" is multiplicative. "Residual" is additive.
            # Assuming Mask * Input (Multiplicative in complex domain) or Additive?
            # Complex multiplication: (a+bi)*(c+di) = (ac-bd) + i(ad+bc).
            # Let's assume Mask is Complex ratio.
            # enhanced_complex = cnn_input * complex_mask (Complex Mul)
            
            m_r = complex_mask[:, 0:1]
            m_i = complex_mask[:, 1:2]
            i_r = coarse_real
            i_i = coarse_imag
            
            final_real = i_r * m_r - i_i * m_i
            final_imag = i_r * m_i + i_i * m_r
            
            # Loss
            # We need Reference Complex (Clean) in the SAME domain (Compressed Mag + Phase)
            ref_real = clean_mag * torch.cos(clean_phase)
            ref_imag = clean_mag * torch.sin(clean_phase)
            ref_complex = torch.cat([ref_real, ref_imag], dim=1)
            
            est_complex = torch.cat([final_real, final_imag], dim=1)
            
            loss = criterion(enhanced_mag, clean_mag, est_complex, ref_complex)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}", end='\r')
                
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        torch.save({
            'crn': crn.state_dict(),
            'cnn': cnn.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    train()
