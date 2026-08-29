'''
Wav-KAN adapted for ECG Arrhythmia Detection
Based on: Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks
Modified for 1D biomedical signals (PhysioNet ECG)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    Wavelet-based KAN linear layer optimized for 1D signals like ECG
    Key difference from image version: handles 1D input naturally
    """
    def __init__(self, in_features, out_features, wavelet_type='morlet', dropout_rate=0.2):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Learnable wavelet parameters (scale and translation)
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Wavelet basis weights
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Optional base activation (for hybrid approach like Spl-KAN)
        self.weight_base = nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_base, a=math.sqrt(5))

        # Batch normalization (critical for ECG stability)
        self.bn = nn.BatchNorm1d(out_features)
        
        # Dropout for regularization (important for ECG with limited data)
        self.dropout = nn.Dropout(dropout_rate)

    def wavelet_transform(self, x):
        """
        Apply wavelet transformation to input signal
        Input x shape: [batch_size, in_features] for 1D signals
        Output: [batch_size, out_features]
        """
        # Expand for broadcasting: [batch, in_features, 1] -> [batch, out_features, in_features]
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)  # [batch, 1, in_features]
        else:
            x_expanded = x

        # Normalize input using learnable scale and translation
        # This is crucial for ECG to handle different signal amplitudes
        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / (scale_expanded + 1e-8)  # Add epsilon for stability

        # Apply different wavelet types
        if self.wavelet_type == 'morlet':
            # Best for ECG: captures frequency content with oscillatory behavior
            # Good for detecting QRS complexes and arrhythmia patterns
            omega0 = 5.0  # Central frequency (tune for ECG: higher=sharper peaks)
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'mexican_hat':
            # Also good for ECG: detects rapid changes (edges) in signal
            # Useful for P-wave, QRS complex, T-wave detection
            term1 = (x_scaled ** 2) - 1
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'dog':
            # Derivative of Gaussian: good for edge detection in ECG
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'meyer':
            # Smooth wavelets with good frequency localization
            v = torch.abs(x_scaled)
            pi = math.pi

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)

            def meyer_aux(v):
                return torch.where(
                    v <= 1/2, 
                    torch.ones_like(v), 
                    torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1)))
                )

            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        return wavelet_output

    def forward(self, x):
        """
        Forward pass combining wavelet and base activation
        """
        wavelet_output = self.wavelet_transform(x)
        
        # Optional: add base activation component (uncomment to enable Spl-KAN style)
        # base_output = F.linear(x, self.weight_base)
        # combined_output = wavelet_output + base_output
        
        combined_output = wavelet_output
        
        # Batch normalization
        output = self.bn(combined_output)
        
        # Dropout for regularization
        output = self.dropout(output)
        
        return output


class ECG_KAN(nn.Module):
    """
    Wav-KAN model for ECG classification
    Specifically designed for 1D biomedical signal processing
    """
    def __init__(self, input_size, hidden_sizes, num_classes, wavelet_type='morlet', dropout_rate=0.2):
        """
        Args:
            input_size: Length of ECG signal (e.g., 5000 for 50 seconds at 100 Hz)
            hidden_sizes: List of hidden layer dimensions (e.g., [256, 128, 64])
            num_classes: Number of arrhythmia classes (typically 5)
            wavelet_type: Type of wavelet ('morlet', 'mexican_hat', 'dog', 'meyer')
            dropout_rate: Dropout probability for regularization
        """
        super(ECG_KAN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.wavelet_type = wavelet_type
        
        # Build layers
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [num_classes]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                KANLinear(layer_sizes[i], layer_sizes[i+1], wavelet_type, dropout_rate)
            )
    
    def forward(self, x):
        """
        Forward pass through ECG_KAN
        Input: [batch_size, signal_length]
        Output: [batch_size, num_classes]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation after each layer except the last
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# Example usage and model initialization
if __name__ == "__main__":
    # ECG parameters
    signal_length = 5000  # 50 seconds at 100 Hz sampling
    num_arrhythmia_classes = 5  # Normal, AFIB, PVC, LBBB, RBBB (example)
    
    # Initialize model
    model = ECG_KAN(
        input_size=signal_length,
        hidden_sizes=[256, 128, 64],
        num_classes=num_arrhythmia_classes,
        wavelet_type='morlet',
        dropout_rate=0.2
    )
    
    # Test forward pass
    dummy_ecg = torch.randn(32, signal_length)  # Batch of 32 ECG signals
    output = model(dummy_ecg)
    
    print(f"Input shape: {dummy_ecg.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
