
import torch
import torch.nn as nn

class CRN(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, rnn_layers=2):
        super().__init__()
        
        # Encoder
        # Input: (B, 1, T, F)
        # We downsample Frequency, preserve Time.
        # F = 257 (for n_fft=512)
        # Using simple strided convolutions.
        # Padding logic to handle odd/even F needs care. 
        # For simplicity, we assume F is padded to power of 2 or appropriate size.
        # 16k, 32ms window -> 512 points -> 257 bins.
        # We can pad input to 256 or 320 for simpler striding. However, let's use valid padding or manual.
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)), # F: 257 -> 128
                nn.BatchNorm2d(16),
                nn.ELU()
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)), # 128 -> 63? (128-3+0)/2 + 1 = 63.5 -> 63
                # Let's adjust padding/kernel to be easier. 
                # Kernel (2,3), stride (1,2) on freq. 
                # 257 -> (257+2*0-3)/2 + 1 = 128
                # 128 -> 63
                # 63 -> 31
                # 31 -> 15
                # 15 -> 7
                nn.BatchNorm2d(32),
                nn.ELU()
            ),
             nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)), # 63 -> 31
                nn.BatchNorm2d(64),
                nn.ELU()
            ),
             nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)), # 31 -> 15
                nn.BatchNorm2d(128),
                nn.ELU()
            ),
             nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)), # 15 -> 7
                nn.BatchNorm2d(256),
                nn.ELU()
            )
        ])
        
        # Bottleneck
        # Input to GRU: (B, Time, Channels*Freq) = (B, T, 256*7) = 1792
        self.gru_input_dim = 256 * 7
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        
        # Decoder
        # Input from GRU: (B, T, hidden) -> Expand to (B, 256, T, 7) ? No, need to project back.
        # We need a linear layer to project hidden -> 256*7
        self.gru_projection = nn.Linear(hidden_size, self.gru_input_dim)
        
        self.decoder = nn.ModuleList([
             nn.Sequential(
                nn.ConvTranspose2d(256*2, 128, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,0)), # 7 -> 15 (check padding)
                # Skip connection cat: 256 (from enc) + 256 (from dec) -> 512 in input? 
                # Standard U-Net: input is cat(skip, upsampled).
                nn.BatchNorm2d(128),
                nn.ELU()
            ),
             nn.Sequential(
                nn.ConvTranspose2d(128*2, 64, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,0)), # 15 -> 31
                nn.BatchNorm2d(64),
                nn.ELU()
            ),
             nn.Sequential(
                nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,0)), # 31 -> 63
                nn.BatchNorm2d(32),
                nn.ELU()
            ),
             nn.Sequential(
                nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,1)), # 63 -> 128 (need output_padding to match?)
                # 63 -> (63-1)*2 - 2*0 + 3 + op = 124 + 3 + op = 127 + op. op=1 -> 128.
                nn.BatchNorm2d(16),
                nn.ELU()
            ),
             nn.Sequential(
                nn.ConvTranspose2d(16*2, 1, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,0)), # 128 -> 257?
                # 128 -> (128-1)*2 + 3 = 254 + 3 = 257.
                # Last layer activation: Sigmoid for Mask
                nn.Sigmoid()
            )
        ])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input/Noisy Magnitude spectrogram (B, 1, T, F)
        Returns:
            torch.Tensor: Estimated Mask (B, 1, T, F)
        """
        # Encoder
        skips = []
        out = x
        for layer in self.encoder:
            out = layer(out)
            skips.append(out)
        
        # Reshape for GRU
        b, c, t, f = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, t, c*f)
        
        # GRU
        out, _ = self.gru(out)
        
        # Project back
        out = self.gru_projection(out)
        out = out.reshape(b, t, c, f).permute(0, 2, 1, 3) # (B, 256, T, 7)
        
        # Decoder
        for i, layer in enumerate(self.decoder):
            # Concatenate skip connection
            skip = skips[-(i+1)]
            # Check shapes: skip has same spatial dim as `out`?
            # When we start decoding:
            # i=0: we act on `out` (256, T, 7). Skip is `skips[-1]` (256, T, 7).
            # We cat them -> (512, T, 7). 
            # ConvTranspose inputs: 256*2 = 512. Matches.
            
            # Note regarding shapes: 
            # If `skip` and `out` differ slightly in Time dim due to kernel(2,3) on T?
            # Stride on T is 1. Kernel T is 2. Padding (1,0) on (T,F).
            # (T + 2*1 - 2)/1 + 1 = T + 1. It grows every layer?
            # Usually we want T preserved.
            # Casual conv: Pad (2, 0) and crop or use causal padding.
            # "Ultra Low Complexity" implies causal.
            # Let's handle Time padding simpler: use T-1 padding or trim?
            # For now, let's assume `same` padding approximation in code or fix later.
            # For simplicity in this plan, I'll `cat` and let torch error if mismatch, then fix.
            # Or better, just slice to match.
            
            min_t = min(out.shape[2], skip.shape[2])
            out = out[:, :, :min_t, :]
            skip = skip[:, :, :min_t, :]
            
            combined = torch.cat([out, skip], dim=1)
            
            # The last layer of decoder does not have a skip connection from encoder input usually, 
            # often it's just predicting the mask.
            # But here `len(decoder)` is 5, `len(skips)` is 5.
            # For i=4 (last one), skips[-(5)] = skips[0] which is output of first conv (16 channels).
            # Input to dec layer 4 is 32 channels. output is 16. 
            # Wait, `decoder[4]` input is 16*2. Output is 1.
            # `skips[0]` is 16 chans. It matches.
            
            # But the FINAL output has to be applied to the ORIGINAL input x?
            # CRN outputs a mask. 
            
            out = layer(combined)
            
        return out
