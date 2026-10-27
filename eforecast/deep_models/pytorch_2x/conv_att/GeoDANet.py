import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GeographicPositionalEncoding(nn.Module):
    """
    Generates a normalized 2D coordinate grid (Latitude/Longitude)
    and projects it into the feature space.
    """

    def __init__(self, channels, W, H):
        super(GeographicPositionalEncoding, self).__init__()

        # Create normalized coordinate grids from -1.0 to 1.0
        # W maps to Latitude, H maps to Longitude
        w_coords = torch.linspace(-1.0, 1.0, W)
        h_coords = torch.linspace(-1.0, 1.0, H)

        # Create meshgrid using 'ij' indexing (matrix style: row, col)
        grid_w, grid_h = torch.meshgrid(w_coords, h_coords, indexing='ij')

        # Stack into shape: (1, 2, W, H)
        grid = torch.stack((grid_w, grid_h), dim=0).unsqueeze(0)

        # Register as a buffer so it automatically moves to the correct device (CPU/GPU)
        self.register_buffer('grid', grid)

        # 1x1 Convolution to map the 2 coordinate channels up to the feature dimension
        self.coord_proj = nn.Conv2d(2, channels, kernel_size=1)

    def forward(self, x):
        B = x.size(0)

        # Expand the static grid to match the batch size: (B, 2, W, H)
        batch_grid = self.grid.expand(B, -1, -1, -1)

        # Project coordinates into the feature space: (B, C, W, H)
        pos_encoded = self.coord_proj(batch_grid)

        # Add the positional encodings to the input features
        return x + pos_encoded


class PositionAttentionModule(nn.Module):
    """ Spatial (Position) Attention Module """

    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        inter_channels = max(1, in_channels // 8)

        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        # Query: B x N x C'
        proj_query = rearrange(self.query_conv(x), 'b c h w -> b c (h w)').permute(0, 2, 1)
        # Key: B x C' x N
        proj_key = rearrange(self.key_conv(x), 'b c h w -> b c (h w)')

        energy = torch.bmm(proj_query, proj_key)
        energy_max = torch.max(energy, dim=-1, keepdim=True)[0]
        attention = F.softmax(energy - energy_max, dim=-1)

        proj_value = rearrange(self.value_conv(x), 'b c h w -> b c (h w)')
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = rearrange(out, 'b c (h w) -> b c h w', h=W, w=H)

        return self.gamma * out + x


class ChannelAttentionModule(nn.Module):
    """ Channel Attention Module """

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = rearrange(x, 'b c h w -> b c (h w)')
        proj_key = rearrange(x, 'b c h w -> b c (h w)').permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_max = torch.max(energy, dim=-1, keepdim=True)[0]
        attention = F.softmax(energy - energy_max, dim=-1)

        proj_value = rearrange(x, 'b c h w -> b c (h w)')
        out = torch.bmm(attention, proj_value)
        out = rearrange(out, 'b c (h w) -> b c h w', h=W, w=H)

        return self.gamma * out + x


class GeoDualAttentionNetwork(nn.Module):
    """
    Dual Attention Network wrapper for NWP data with Explicit Grid Coordinates
    Input shape: (B, T, D, W, H)
    """

    def __init__(self, shape):
        super(GeoDualAttentionNetwork, self).__init__()
        _, time_steps, variables, W, H = shape
        in_channels = time_steps * variables

        # 1. Geographic Positional Encoding
        self.pos_encoder = GeographicPositionalEncoding(in_channels, W, H)

        # 2. Attention Modules
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()

        # 3. Output Refinement
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, T, D, W, H = x.shape

        # Reshape to (B, T*D, W, H)
        x_reshaped = rearrange(x, 'b t c h w -> b (t c) h w')

        # Inject positional encoding directly into the features
        x_encoded = self.pos_encoder(x_reshaped)

        # Apply Parallel Dual Attention on the structurally aware features
        out_pam = self.pam(x_encoded)
        out_cam = self.cam(x_encoded)

        # Sum and refine
        feat_sum = out_pam + out_cam
        refined_feat = self.conv_out(feat_sum)

        # Reshape back to original 5D tensor format
        out = rearrange(refined_feat, 'b (t c) h w -> b t c h w', t=T, c=D)

        return out
