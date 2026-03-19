import torch
import torch.nn as nn

# -------------------------
# Conformer components
# -------------------------

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        # 🔧 RENAMED: net → sequential
        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.sequential(x)


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )

        self.batchnorm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        x = self.layernorm(x).transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv = ConvModule(d_model, kernel_size)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


def _conv_out_length(length):
    # For kernel=3, stride=2, padding=1
    return (length + 1) // 2


class ConvSubsampling(nn.Module):
    """4x subsampling using two 2D conv layers."""
    def __init__(self, input_dim=80, d_model=256, channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        out_freq = _conv_out_length(_conv_out_length(input_dim))
        self.out = nn.Linear(channels * out_freq, d_model)

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)    # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        t = torch.arange(x.size(1), device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        x_rot = torch.cat(
            (-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]), dim=-1
        )

        return x * cos.unsqueeze(0) + x_rot * sin.unsqueeze(0)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=256, n_layers=6, subsampling=True):
        super().__init__()
        self.subsampling = ConvSubsampling(input_dim, d_model) if subsampling else None
        self.input_projection = None if subsampling else nn.Linear(input_dim, d_model)
        self.rope = RotaryPositionalEmbedding(d_model)

        self.layers = nn.ModuleList(
            [ConformerBlock(d_model) for _ in range(n_layers)]
        )

    def _subsample_lengths(self, lengths):
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths

    def forward(self, x, input_lengths=None):
        if self.subsampling is not None:
            x = self.subsampling(x)
            if input_lengths is not None:
                input_lengths = self._subsample_lengths(input_lengths)
        else:
            x = self.input_projection(x)
        x = self.rope(x)
        for layer in self.layers:
            x = layer(x)
        return x, input_lengths


class ConformerCTC(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, subsampling=True):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=80,
            d_model=d_model,
            n_layers=n_layers,
            subsampling=subsampling
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, input_lengths=None):
        x, out_lengths = self.encoder(x, input_lengths)
        return self.classifier(x), out_lengths
