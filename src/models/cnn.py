
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=2, output_channels=2, hidden_channels=64, num_layers=5):
        """
        Stage 2 CNN for complex mask estimation.
        Input: (B, 2, T, F) representing (Real, Imag) of the coarsely enhanced signal.
        Output: (B, 2, T, F) complex mask (Real, Imag).
        """
        super().__init__()
        
        layers = []
        in_c = input_channels
        
        for i in range(num_layers):
            out_c = hidden_channels if i < num_layers - 1 else output_channels
            
            # Dilated convolutions can help captured context
            dilation = 2**i
            padding = 1 * dilation # For kernel_size 3
            # Or use casual padding. 
            # In frequency (F), we shouldn't dilate or pad freely if F is fixed. 
            # Conv2d over (T, F). 
            # Usually we treat F as dimension and T as sequence. 
            # But Conv2d treats both as spatial.
            # Let's keep kernel (3,3) or (2,3) and use consistent padding.
            
            layers.append(
                nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=1, padding=(1, 1))
            )
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ELU())
            
            in_c = out_c
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Complex spectrogram (B, 2, T, F)
        
        Returns:
            torch.Tensor: Complex Mask (B, 2, T, F)
        """
        return self.net(x)
