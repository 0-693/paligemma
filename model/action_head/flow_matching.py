import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions or time steps."""
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        # Precompute the positional encodings up to max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # Use log scale of frequencies as in Transformer
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        """Returns positional encoding for positions [0, seq_len-1] with shape (seq_len, dim)."""
        if seq_len > self.pe.size(1):
            # Extend the positional encoding if needed
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        """Extend the positional encoding buffer to new_max_len."""
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        # Generate new positional encodings from old_max_len to new_max_len
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        # Concatenate to existing buffer
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    """Linear layer with separate weights for each category (e.g., each robot embodiment or group)."""
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            # Single category uses a standard linear layer
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            # Separate weight and bias for each category
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        """
        x: Tensor of shape (..., in_dim)
        category_id: Tensor of shape (...) with indices of categories for each element in batch.
        """
        if self.num_categories <= 1:
            return self.linear(x)
        # Ensure category_id shape aligns with x batch
        # Flatten batch dims of x for processing
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (B, in_dim)
        if category_id.dim() == 0:
            # Single category for all
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
            # If category_id is given per element in batch
            category_id = category_id.view(-1)  # flatten
            # Batch matrix multiply for category-specific weights
            # (B, 1, in_dim) @ (B, in_dim, out_dim) -> (B, 1, out_dim) -> (B, out_dim)
            weight_selected = self.weight[category_id]        # (B, in_dim, out_dim)
            bias_selected = self.bias[category_id]            # (B, out_dim)
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        # Reshape back to original batch shape with out_dim
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):
    """
    MLP with two CategorySpecificLinear layers (and an activation) for encoding or decoding.
    Can be used for state encoding or action decoding, specialized per category.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        # --- 核心修复：在进入网络前，确保输入张量的数据类型正确 ---
        # 网络的权重是Float32，因此需要将输入转换为Float32
        x = x.to(torch.float32)
        
        # Apply first layer with activation
        out = self.activation(self.fc1(x, category_id))
        # Second layer (linear output, no activation by default)
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):
    """
    Encoder for action sequences that supports multiple embodiments.
    Uses three category-specific linear layers (W1, W2, W3) and sinusoidal position encoding:contentReference[oaicite:6]{index=6}.
    """
    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        # Three linear layers for the encoder as described (W1, W2, W3 all category-specific)
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
        # Positional encoding for sequence positions (length = horizon)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):
        """
        action_seq: Tensor shape [B, H, D] where H = horizon, D = per-step action dimensions (for given embodiment).
        category_id: category indices (size [B]) indicating embodiment type for each sample.
        """
        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
        # Flatten sequence for linear layers
        x = action_seq.reshape(B * H, D)  # shape (B*H, D)
        # Apply first linear layer (W1) with category ids repeated for each time step
        if category_id.dim() == 0:
            # Same category for all, just use one id
            cat_ids = category_id.repeat(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).repeat(1, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            # (B*H, hidden_dim)
        # Add sinusoidal positional encoding (expand to B*H, hidden_dim)
        pos_enc = self.pos_encoding(H).to(out.device)         # shape (1, H, hidden_dim)
        pos_enc = pos_enc.repeat(B, 1, 1).reshape(B * H, -1)  # (B*H, hidden_dim)
        out = out + pos_enc
        out = self.activation(self.W2(out, cat_ids))          # (B*H, hidden_dim)
        out = self.W3(out, cat_ids)                           # (B*H, embed_dim)
        # Reshape back to [B, H, embed_dim]
        out = out.view(B, H, self.embed_dim)
        return out

class BasicTransformerBlock(nn.Module):
    """
    A basic transformer block with cross-attention and feed-forward, conditioned on time embedding.
    This block attends action tokens (queries) to context tokens (keys/values), then applies a feed-forward network.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network for transformer block
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):
        # action_tokens: [B, H, embed_dim], context_tokens: [B, T, embed_dim], time_emb: [B, embed_dim]
        # Cross-Attention: queries = action_tokens, keys=values = context_tokens
        x = self.norm1(action_tokens)
        # MultiheadAttention expects (batch, seq, dim) with batch_first=True (PyTorch 1.12+)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)
        # Residual connection
        x = action_tokens + attn_out
        # Feed-forward with time embedding injection
        x2 = self.norm2(x)
        # If a time embedding is provided, add it (broadcast across sequence length)
        if time_emb is not None:
            # time_emb shape [B, embed_dim] -> [B, 1, embed_dim] to add to each token
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):
    """
    Flow-matching based action prediction head.
    It predicts continuous actions from fused image-language tokens using a learned vector field (velocity).
    Supports multi-embodiment via category-specific sub-modules and iterative Euler integration for inference.
    """
    def __init__(self, config=None,
                 embed_dim: int = 1536,
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 50,
                 num_categories: int = 1):
        super().__init__()
        # If config provided, override defaults
        if config is not None:
            # Assume config is a dict or object with attributes
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            # Create a simple config namespace to store parameters
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories)
        # Sub-modules
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = config.per_action_dim
        self.action_dim = config.action_dim

        # Positional encoder for continuous time (diffusion timestep embedding)
        # We'll use a simple sinusoidal encoding for time as well (1D sequence of length=1)
        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)
        # Transformer blocks for cross-attention
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])
        # LayerNorm for transformer output
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        # MLP head to predict action velocity (or action output) from transformer output tokens
        # We use CategorySpecificMLP to allow different output mappings per embodiment category.
        # The input to mlp_head is a pooled representation of action tokens (e.g., last token or mean).
        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)
        # Optional state encoder (for robot states) and action encoder for sequences:
        # If state input is used, initialize a state encoder to embed it into embed_dim.
        # If action horizon > 1, initialize action_encoder to embed sequences.
        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
            # state_dim and state_hidden can be part of config if needed
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
            # Use CategorySpecificMLP to encode state vector to embed_dim
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)
        # Action sequence encoder (if horizon > 1)
        self.action_encoder = None
        if horizon > 1:
            # per-step action dimension might be provided in config (e.g., config.per_action_dim)
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
                # Assume action_dim is total = horizon * per_step_dim if per_action_dim not given
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  # hidden_dim for encoder set equal to embed_dim for simplicity
                                                               horizon=horizon,
                                                               num_categories=num_categories)

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None):
        """
        Forward pass for training. If `actions_gt` (ground-truth actions) is provided, 
        computes the predicted velocity or noise for a random time step and returns it for loss calculation.
        If `actions_gt` is None, it will perform inference and return predicted actions (calls get_action).
        
        fused_tokens: [B, T, embed_dim] fused image-language (and possibly state) tokens.
        state: [B, S] optional robot state vector (S depends on embodiment).
        actions_gt: [B, action_dim] ground truth action vector or sequence (flattened if sequence).
        embodiment_id: [B] indices for robot embodiment category (if multiple); None if single embodiment.
        """
        # If no ground truth provided, run inference to get action prediction
        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id)
        B = fused_tokens.size(0)
        device = fused_tokens.device
        # If multiple categories and no id provided, default to 0 for all
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        # Encode state if provided, and append to context tokens
        context_tokens = fused_tokens  # shape [B, T, embed_dim]
        if state is not None and self.state_encoder is not None:
            # Encode current state to embedding token
            state_emb = self.state_encoder(state, embodiment_id)  # [B, embed_dim]
            state_emb = state_emb.unsqueeze(1)  # as one token
            # Concatenate state token to context tokens
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)  # [B, T+1, embed_dim]
        # Sample random time t (Beta or uniform) for training
        # t = torch.rand(B, device=device)  # Uniform(0,1)
        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)

        
                            
        # If Beta noise modeling is desired, one could sample t from Beta distribution, e.g.:
        # t = torch.betainc(alpha=0.5, beta=0.5, size=(B,), device=device)  # symmetric Beta(0.5,0.5) as example
        # For simplicity using uniform here.
        # Prepare time embedding
        # Use a positional encoding for time (scale continuous t to index range). We scale t by max_len for encoding index.
        time_index = (t * 1000).long()  # convert t to an index (0 to 999)
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0)  # [B, embed_dim]
        # Sample initial noise from simple distribution (e.g., Normal or Uniform in [-1,1])
        action_shape = actions_gt.shape[1]  # total action dim (flattened sequence)
        # Here we assume actions_gt is flattened if sequence. We will reshape if needed:

        actions_gt_seq = actions_gt  # already [B, H, D]

            
        # Define initial noise distribution (Beta noise modeling implies using bounded noise).
        # We'll use uniform noise in [-1,1] for each action dimension for simplicity.
        noise = torch.rand_like(actions_gt) * 2 - 1  # Uniform(-1,1) same shape as actions_gt
        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
        else:
            noise_seq = noise.unsqueeze(1)
        # Compute the interpolated action at time t: A_t = (1 - t)*noise + t*target (linear interpolation path:contentReference[oaicite:7]{index=7})
        # We broadcast t to shape [B, horizon, 1] for sequence or [B, 1] for single
        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  # shape [B, H, per_action_dim]
        # Encode the current intermediate action sequence into tokens (if horizon > 1)
        if self.horizon > 1 and self.action_encoder is not None:
            # Embed the sequence of actions to token representation
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  # [B, H, embed_dim]
        else:
            # If horizon == 1, we still create a token of shape [B, 1, embed_dim] representing the action value
            # We can project the single action to embed_dim with a linear layer, but here directly expand to embed_dim:
            # (Alternatively, could treat action value as a "token" by duplicating it to embed_dim via linear mapping)
            # For simplicity, initialize a learned linear mapping for single action token if needed.
            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq)  # [B, 1, embed_dim]
        # Pass through transformer blocks with cross-attention
        x = action_tokens  # [B, H, embed_dim]
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        # Normalize output
        x = self.norm_out(x)  # [B, H, embed_dim]
        # Pool or select token representation for output:
        if self.horizon > 1:
            # If predicting a sequence, we could output velocity for each time step.
            # Here we output the velocity for the entire sequence flattened (target - noise).
            # We flatten tokens and use mlp_head to produce flattened delta.
            x_flat = x.reshape(B, -1)  # [B, H*embed_dim]
            # If needed, apply a linear down-projection before mlp_head for dimensionality alignment
            if not hasattr(self, "seq_pool_proj"):
                # Project H*embed_dim to embed_dim for MLP head input
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
            x_pooled = self.seq_pool_proj(x_flat)  # [B, embed_dim]
        else:
            # If single action token, just take the token itself (of shape [B, 1, embed_dim])
            x_pooled = x.squeeze(1)  # [B, embed_dim]
        # Predict velocity (difference between target and noise) in original action space
        pred_velocity = self.mlp_head(x_pooled, embodiment_id)  # [B, action_dim]
        # Return the predicted velocity (to be supervised with (actions_gt - noise) or similar)
        return pred_velocity, noise

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None):
        """
        Inference: generate action by integrating the learned flow (Euler method).
        """
        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        # Prepare context tokens (include state if provided)
        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)  # [B, 1, embed_dim]
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)
        # Initialize action sequence from noise (e.g. uniform random in [-1,1])
        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
            # If not set, infer from mlp_head output dimension
            action_dim_total = self.action_dim
        # Determine per-step action dimension
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total
        # Sample initial noise for action(s)
        # If action range is known, sample accordingly (here assuming [-1,1] for each dof)
        action = (torch.rand(B, action_dim_total, device=device) * 2 - 1)
        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)
        # Euler integration over num_inference_timesteps
        N = int(getattr(self.config, "num_inference_timesteps", 50))
        dt = 1.0 / N
        for i in range(N):
            t = i / N  # current time fraction
            # Prepare time embedding for current step
            time_index = int(t * 1000)
            time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0)  # [embed_dim]
            time_emb = time_emb.unsqueeze(0).repeat(B, 1)  # [B, embed_dim]
            # Encode current action sequence to tokens
            if self.horizon > 1 and self.action_encoder is not None:
                action_tokens = self.action_encoder(action_seq, embodiment_id)  # [B, H, embed_dim]
            else:
                if hasattr(self, "single_action_proj"):
                    action_tokens = self.single_action_proj(action_seq)  # [B, 1, embed_dim]
                else:
                    # If single action and no proj, create one on the fly
                    self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                    action_tokens = self.single_action_proj(action_seq)
            # Pass through transformer blocks
            x = action_tokens
            for block in self.transformer_blocks:
                x = block(x, context_tokens, time_emb)
            x = self.norm_out(x)
            # Pool tokens
            if self.horizon > 1:
                x_flat = x.reshape(B, -1)
                if hasattr(self, "seq_pool_proj"):
                    x_pooled = self.seq_pool_proj(x_flat)
                else:
                    # create on the fly if not present
                    self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                    x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)
            # Predict velocity in action space
            pred = self.mlp_head(x_pooled, embodiment_id)  # [B, action_dim_total]
            # Update action using Euler step: new_action = old_action + dt * pred:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
            action = action + dt * pred
            # Reshape updated action into sequence form for next iteration
            if self.horizon > 1:
                action_seq = action.view(B, self.horizon, per_action_dim)
            else:
                action_seq = action.view(B, 1, per_action_dim)
        # After integration, `action` contains the final predicted action vector [B, action_dim_total]
        return action

    @property
    def device(self):
        # Convenience to get device of module
        return next(self.parameters()).device

    @property
    def dtype(self):
        # Convenience to get dtype of module parameters
        return next(self.parameters()).dtype

