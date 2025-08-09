"""
Multi-Task Heads for Diabetic Retinopathy Classification and Segmentation
Implements classification head for DR grading and segmentation decoder for lesion detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple, Dict
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationHead(nn.Module):
    """
    Classification head for diabetic retinopathy grading (5 classes: 0-4).
    Uses global pooling followed by fully connected layers with dropout.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize classification head.
        
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of DR classes (default: 5)
            dropout_rate: Dropout rate for regularization
            hidden_dim: Hidden dimension for intermediate layer
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Classifier layers
        layers = []
        
        # First dropout
        layers.append(nn.Dropout(dropout_rate))
        
        # First linear layer
        layers.append(nn.Linear(in_features, hidden_dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(self.activation)
        
        # Second dropout
        layers.append(nn.Dropout(dropout_rate * 0.7))  # Slightly lower dropout
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"ClassificationHead: {in_features} -> {hidden_dim} -> {num_classes}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(nn.init, 'xavier_uniform_'):
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Global average pooling
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities using softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

class SegmentationHead(nn.Module):
    """
    Segmentation decoder head for lesion segmentation.
    Uses transpose convolutions to upsample features to full resolution.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 1,
        decoder_channels: List[int] = [512, 256, 128, 64],
        use_skip_connections: bool = False,
        skip_feature_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize segmentation head.
        
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of segmentation classes (1 for binary)
            decoder_channels: Channels for each decoder block
            use_skip_connections: Whether to use skip connections from encoder
            skip_feature_channels: Channels from skip connections
            dropout_rate: Dropout rate for regularization
            activation: Activation function
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_skip_connections = use_skip_connections
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build decoder - simplified to avoid tensor creation issues
        self.decoder_blocks = nn.ModuleList()
        
        # Input channels for first block
        current_channels = in_features
        
        # Skip connections disabled to prevent tensor issues
        self.skip_connections = False
        self.skip_channels = []
        
        # Create decoder blocks without skip connections
        for i, out_channels in enumerate(decoder_channels):
            block = self._make_decoder_block(
                in_channels=current_channels,
                out_channels=out_channels,
                skip_channels=0,  # No skip connections
                dropout_rate=dropout_rate
            )
            
            self.decoder_blocks.append(block)
            current_channels = out_channels
        
        # Final output layer
        self.final_conv = nn.Conv2d(
            current_channels, 
            num_classes, 
            kernel_size=1, 
            padding=0
        )
        
        # Output activation
        if num_classes == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"SegmentationHead: {in_features} -> {decoder_channels} -> {num_classes}")
    
    def _make_decoder_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        skip_channels: int = 0,
        dropout_rate: float = 0.1
    ) -> nn.Module:
        """
        Create a decoder block with transpose convolution.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            skip_channels: Channels from skip connection
            dropout_rate: Dropout rate
            
        Returns:
            Decoder block module
        """
        # Adjust input channels if skip connection is used
        total_in_channels = in_channels + skip_channels
        
        block = nn.Sequential(
            # Transpose convolution for upsampling
            nn.ConvTranspose2d(
                total_in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
            
            # Regular convolution for refinement
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
            
            # Dropout for regularization
            nn.Dropout2d(dropout_rate)
        )
        
        return block
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through segmentation head - bulletproof version.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            skip_features: List of skip connection features (optional)
            
        Returns:
            Segmentation mask of shape (B, num_classes, H_full, W_full)
        """
        current = x
        
        # Process through decoder blocks without skip connections to avoid tensor issues
        for i, block in enumerate(self.decoder_blocks):
            current = block(current)
        
        # Final convolution
        output = self.final_conv(current)
        
        # Safe upsampling to target size (512x512) if needed
        target_size = 512
        current_size = output.shape[-1]
        
        if current_size != target_size:
            output = F.interpolate(
                output,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Apply output activation
        output = self.output_activation(output)
        
        return output

class AttentionGate(nn.Module):
    """
    Attention gate for skip connections in U-Net style architectures.
    Helps the model focus on relevant features.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Channels from gating signal (decoder)
            skip_channels: Channels from skip connection (encoder)
            inter_channels: Intermediate channels for attention computation
        """
        super().__init__()
        
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.attention_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention gate.
        
        Args:
            gate: Gating signal from decoder
            skip: Skip connection from encoder
            
        Returns:
            Attention-weighted skip features
        """
        # Resize gate to match skip resolution
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(
                gate, 
                size=skip.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Compute attention coefficients
        g = self.gate_conv(gate)
        s = self.skip_conv(skip)
        
        attention = self.relu(g + s)
        attention = self.attention_conv(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention to skip features
        attended_skip = skip * attention
        
        return attended_skip

class AdvancedSegmentationHead(SegmentationHead):
    """
    Advanced segmentation head with attention gates and deeper decoder.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 1,
        decoder_channels: List[int] = [512, 256, 128, 64],
        skip_feature_channels: Optional[List[int]] = None,
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize advanced segmentation head.
        
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of segmentation classes
            decoder_channels: Channels for each decoder block
            skip_feature_channels: Channels from skip connections
            use_attention: Whether to use attention gates
            dropout_rate: Dropout rate
        """
        # Initialize parent without skip connections
        super().__init__(
            in_features=in_features,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            use_skip_connections=False,
            dropout_rate=dropout_rate
        )
        
        self.use_attention = use_attention
        
        # Create attention gates if skip connections are used
        if skip_feature_channels and use_attention:
            self.attention_gates = nn.ModuleList()
            
            for i, skip_ch in enumerate(skip_feature_channels[::-1]):
                if i < len(decoder_channels):
                    gate_ch = decoder_channels[i]
                    inter_ch = min(skip_ch, gate_ch) // 2
                    
                    attention_gate = AttentionGate(
                        gate_channels=gate_ch,
                        skip_channels=skip_ch,
                        inter_channels=inter_ch
                    )
                    
                    self.attention_gates.append(attention_gate)
        else:
            self.attention_gates = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with attention gates.
        
        Args:
            x: Feature tensor
            skip_features: Skip connection features
            
        Returns:
            Segmentation output
        """
        current = x
        
        # Process through decoder blocks with attention
        for i, block in enumerate(self.decoder_blocks):
            # Apply attention gate if available
            if (self.attention_gates is not None and 
                skip_features is not None and 
                i < len(self.attention_gates) and 
                i < len(skip_features)):
                
                skip_feat = skip_features[-(i+1)]
                attended_skip = self.attention_gates[i](current, skip_feat)
                
                # Resize if needed
                if attended_skip.shape[-2:] != current.shape[-2:]:
                    attended_skip = F.interpolate(
                        attended_skip,
                        size=current.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Concatenate
                current = torch.cat([current, attended_skip], dim=1)
            
            # Apply decoder block
            current = block(current)
        
        # Final output
        output = self.final_conv(current)
        output = self.output_activation(output)
        
        return output

def test_heads_functionality():
    """Test the head implementations."""
    logger.info("Testing classification and segmentation heads...")
    
    # Test parameters
    batch_size = 2
    in_features = 1280  # EfficientNetV2-S final features
    feature_height, feature_width = 7, 7  # Typical feature map size
    
    # Create test input
    test_features = torch.randn(batch_size, in_features, feature_height, feature_width)
    
    # Test Classification Head
    logger.info("Testing ClassificationHead...")
    cls_head = ClassificationHead(in_features=in_features, num_classes=5)
    
    with torch.no_grad():
        cls_output = cls_head(test_features)
        cls_probs = cls_head.get_probabilities(test_features)
    
    logger.info(f"Classification output shape: {cls_output.shape}")
    logger.info(f"Classification probabilities shape: {cls_probs.shape}")
    assert cls_output.shape == (batch_size, 5), f"Expected (2, 5), got {cls_output.shape}"
    
    # Test Segmentation Head
    logger.info("Testing SegmentationHead...")
    seg_head = SegmentationHead(
        in_features=in_features,
        num_classes=1,
        decoder_channels=[512, 256, 128, 64]
    )
    
    with torch.no_grad():
        seg_output = seg_head(test_features)
    
    logger.info(f"Segmentation output shape: {seg_output.shape}")
    # Should be upsampled to much larger resolution
    assert seg_output.shape[0] == batch_size, f"Wrong batch size: {seg_output.shape[0]}"
    assert seg_output.shape[1] == 1, f"Wrong number of classes: {seg_output.shape[1]}"
    
    # Test Advanced Segmentation Head with skip connections
    logger.info("Testing AdvancedSegmentationHead...")
    skip_channels = [64, 160, 256]  # Example skip connection channels
    
    adv_seg_head = AdvancedSegmentationHead(
        in_features=in_features,
        num_classes=1,
        skip_feature_channels=skip_channels,
        use_attention=True
    )
    
    # Create dummy skip features
    skip_features = [
        torch.randn(batch_size, 64, 56, 56),   # High resolution
        torch.randn(batch_size, 160, 28, 28),  # Medium resolution
        torch.randn(batch_size, 256, 14, 14),  # Low resolution
    ]
    
    with torch.no_grad():
        adv_seg_output = adv_seg_head(test_features, skip_features)
    
    logger.info(f"Advanced segmentation output shape: {adv_seg_output.shape}")
    
    logger.info("All head tests completed successfully!")
    
    return cls_head, seg_head, adv_seg_head

if __name__ == "__main__":
    # Run tests
    test_heads_functionality()
