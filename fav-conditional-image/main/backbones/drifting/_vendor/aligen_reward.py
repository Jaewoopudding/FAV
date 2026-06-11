"""
JAX implementation of the aesthetic reward model (CLIP ViT-L/14 + MLP).

Provides differentiable reward computation for ALIGEN training on TPU/GPU via JAX.
CLIP weights are loaded from HuggingFace PyTorch model and converted to JAX arrays.
MLP weights are loaded from the pretrained aesthetic scorer checkpoint.
"""
import jax
import jax.numpy as jnp
import numpy as np

# ─── CLIP ViT-L/14 constants ────────────────────────────────────────────
CLIP_HIDDEN_SIZE = 1024
CLIP_NUM_HEADS = 16
CLIP_NUM_LAYERS = 24
CLIP_INTERMEDIATE_SIZE = 4096
CLIP_PATCH_SIZE = 14
CLIP_IMAGE_SIZE = 224
CLIP_PROJECTION_DIM = 768
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ─── Elementary ops ──────────────────────────────────────────────────────

def quick_gelu(x):
    """CLIP uses this approximation instead of standard GELU."""
    return x * jax.nn.sigmoid(1.702 * x)


def layer_norm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * weight + bias


# ─── CLIP vision encoder (functional) ───────────────────────────────────

def clip_attention(x, params, num_heads=CLIP_NUM_HEADS):
    """Multi-head self-attention (no causal mask for vision)."""
    B, N, D = x.shape
    head_dim = D // num_heads

    q = x @ params['q_w'] + params['q_b']
    k = x @ params['k_w'] + params['k_b']
    v = x @ params['v_w'] + params['v_b']

    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    scale = jnp.float32(head_dim) ** -0.5
    attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.matmul(attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
    return out @ params['out_w'] + params['out_b']


@jax.checkpoint
def clip_encoder_layer(x, params):
    """One CLIP ViT encoder layer (gradient-checkpointed to save memory)."""
    # Self-attention block
    h = layer_norm(x, params['ln1_w'], params['ln1_b'])
    h = clip_attention(h, params)
    x = x + h

    # MLP block
    h = layer_norm(x, params['ln2_w'], params['ln2_b'])
    h = h @ params['fc1_w'] + params['fc1_b']
    h = quick_gelu(h)
    h = h @ params['fc2_w'] + params['fc2_b']
    x = x + h
    return x


def clip_vision_forward(pixel_values, clip_params):
    """
    CLIP ViT-L/14 vision encoder: images -> L2-normalised 768-d embeddings.

    Args:
        pixel_values: (B, H, W, C) float32 images in [-1, 1] (BHWC, JAX convention).
        clip_params:  dict of JAX arrays produced by ``load_clip_params_jax``.

    Returns:
        (B, 768) L2-normalised CLIP image embeddings.
    """
    B = pixel_values.shape[0]

    # Preprocess: [-1,1] -> [0,1] -> resize 224 -> CLIP-normalise
    x = jnp.clip((pixel_values + 1) / 2, 0, 1)
    x = jax.image.resize(x, (B, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, 3), method='bilinear')
    x = (x - CLIP_MEAN) / CLIP_STD

    # Patch embedding (Conv2D, stride=14)
    patches = jax.lax.conv_general_dilated(
        x,
        clip_params['patch_w'],                    # (14, 14, 3, 1024)
        window_strides=(CLIP_PATCH_SIZE, CLIP_PATCH_SIZE),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )                                               # (B, 16, 16, 1024)
    patches = patches.reshape(B, -1, CLIP_HIDDEN_SIZE)  # (B, 256, 1024)

    # Prepend [CLS] token + positional embedding
    cls = jnp.broadcast_to(clip_params['class_emb'], (B, 1, CLIP_HIDDEN_SIZE))
    x = jnp.concatenate([cls, patches], axis=1)     # (B, 257, 1024)
    x = x + clip_params['pos_emb']

    # Pre-LayerNorm
    x = layer_norm(x, clip_params['pre_ln_w'], clip_params['pre_ln_b'])

    # 24 transformer blocks
    for i in range(CLIP_NUM_LAYERS):
        x = clip_encoder_layer(x, clip_params[f'layer_{i}'])

    # [CLS] token + post-LayerNorm
    cls_out = x[:, 0]
    cls_out = layer_norm(cls_out, clip_params['post_ln_w'], clip_params['post_ln_b'])

    # Visual projection -> 768-d
    features = cls_out @ clip_params['proj_w']

    # L2 normalise
    features = features / (jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    return features


# ─── MLP aesthetic scorer (functional) ───────────────────────────────────

def mlp_diff_forward(features, mlp_params):
    """
    MLP aesthetic score predictor (eval mode - no dropout).

    Architecture: 768 -> 1024 -> 128 -> 64 -> 16 -> 1
    Note: there are NO activation functions between layers (purely linear).

    Args:
        features: (B, 768) CLIP features.
        mlp_params: dict from ``load_mlp_params_jax``.

    Returns:
        (B,) aesthetic scores.
    """
    x = features
    x = x @ mlp_params['w0'] + mlp_params['b0']   # 768  -> 1024
    x = x @ mlp_params['w1'] + mlp_params['b1']   # 1024 -> 128
    x = x @ mlp_params['w2'] + mlp_params['b2']   # 128  -> 64
    x = x @ mlp_params['w3'] + mlp_params['b3']   # 64   -> 16
    x = x @ mlp_params['w4'] + mlp_params['b4']   # 16   -> 1
    return x.squeeze(-1)


def aesthetic_score(images, clip_params, mlp_params):
    """End-to-end: images (BHWC, [-1,1]) -> (B,) aesthetic scores."""
    features = clip_vision_forward(images, clip_params)
    return mlp_diff_forward(features, mlp_params)


# ─── Weight loading utilities ────────────────────────────────────────────

def load_clip_params_jax(model_name="openai/clip-vit-large-patch14"):
    """Load CLIP ViT-L/14 *vision-encoder* weights from HuggingFace and convert to JAX.

    Requires ``torch`` and ``transformers`` (CPU-only is fine).
    """
    import torch
    from transformers import CLIPModel

    print(f"[aligen_reward] Loading CLIP weights from {model_name} ...")
    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float32)
    sd = model.state_dict()

    params = {}

    # Patch embedding: PyTorch (out, in, H, W) -> JAX (H, W, in, out)
    params['patch_w'] = jnp.array(sd['vision_model.embeddings.patch_embedding.weight']
                                   .permute(2, 3, 1, 0).numpy())

    # Class token: (1024,) -> (1, 1, 1024) for broadcast
    params['class_emb'] = jnp.array(sd['vision_model.embeddings.class_embedding']
                                     .numpy())[None, None, :]

    # Position embedding: (257, 1024) -> (1, 257, 1024)
    params['pos_emb'] = jnp.array(sd['vision_model.embeddings.position_embedding.weight']
                                   .numpy())[None, :, :]

    # Pre-LayerNorm
    params['pre_ln_w'] = jnp.array(sd['vision_model.pre_layrnorm.weight'].numpy())
    params['pre_ln_b'] = jnp.array(sd['vision_model.pre_layrnorm.bias'].numpy())

    # Encoder layers
    for i in range(CLIP_NUM_LAYERS):
        pfx = f'vision_model.encoder.layers.{i}'
        lp = {}
        # Self-attention (transpose linear weights: (out, in) -> (in, out))
        lp['q_w']   = jnp.array(sd[f'{pfx}.self_attn.q_proj.weight'].T.numpy())
        lp['q_b']   = jnp.array(sd[f'{pfx}.self_attn.q_proj.bias'].numpy())
        lp['k_w']   = jnp.array(sd[f'{pfx}.self_attn.k_proj.weight'].T.numpy())
        lp['k_b']   = jnp.array(sd[f'{pfx}.self_attn.k_proj.bias'].numpy())
        lp['v_w']   = jnp.array(sd[f'{pfx}.self_attn.v_proj.weight'].T.numpy())
        lp['v_b']   = jnp.array(sd[f'{pfx}.self_attn.v_proj.bias'].numpy())
        lp['out_w'] = jnp.array(sd[f'{pfx}.self_attn.out_proj.weight'].T.numpy())
        lp['out_b'] = jnp.array(sd[f'{pfx}.self_attn.out_proj.bias'].numpy())
        # Layer norms
        lp['ln1_w'] = jnp.array(sd[f'{pfx}.layer_norm1.weight'].numpy())
        lp['ln1_b'] = jnp.array(sd[f'{pfx}.layer_norm1.bias'].numpy())
        lp['ln2_w'] = jnp.array(sd[f'{pfx}.layer_norm2.weight'].numpy())
        lp['ln2_b'] = jnp.array(sd[f'{pfx}.layer_norm2.bias'].numpy())
        # MLP
        lp['fc1_w'] = jnp.array(sd[f'{pfx}.mlp.fc1.weight'].T.numpy())
        lp['fc1_b'] = jnp.array(sd[f'{pfx}.mlp.fc1.bias'].numpy())
        lp['fc2_w'] = jnp.array(sd[f'{pfx}.mlp.fc2.weight'].T.numpy())
        lp['fc2_b'] = jnp.array(sd[f'{pfx}.mlp.fc2.bias'].numpy())
        params[f'layer_{i}'] = lp

    # Post-LayerNorm
    params['post_ln_w'] = jnp.array(sd['vision_model.post_layernorm.weight'].numpy())
    params['post_ln_b'] = jnp.array(sd['vision_model.post_layernorm.bias'].numpy())

    # Visual projection: (768, 1024) -> (1024, 768)
    params['proj_w'] = jnp.array(sd['visual_projection.weight'].T.numpy())

    del model, sd
    print("[aligen_reward] CLIP weights loaded and converted to JAX.")
    return params


def load_mlp_params_jax(weights_path):
    """Load MLP aesthetic-scorer weights from a PyTorch ``.pth`` file.

    The architecture is ``nn.Sequential(Linear(768,1024), Dropout, Linear(1024,128),
    Dropout, Linear(128,64), Dropout, Linear(64,16), Linear(16,1))``.
    Sequential indices: 0(linear), 1(dropout), 2(linear), 3(dropout),
    4(linear), 5(dropout), 6(linear), 7(linear).
    """
    import torch
    print(f"[aligen_reward] Loading MLP weights from {weights_path} ...")
    sd = torch.load(weights_path, map_location='cpu')

    # Transpose every weight: PyTorch (out, in) -> JAX (in, out)
    params = {
        'w0': jnp.array(sd['layers.0.weight'].T.numpy()),   # 768  -> 1024
        'b0': jnp.array(sd['layers.0.bias'].numpy()),
        'w1': jnp.array(sd['layers.2.weight'].T.numpy()),   # 1024 -> 128
        'b1': jnp.array(sd['layers.2.bias'].numpy()),
        'w2': jnp.array(sd['layers.4.weight'].T.numpy()),   # 128  -> 64
        'b2': jnp.array(sd['layers.4.bias'].numpy()),
        'w3': jnp.array(sd['layers.6.weight'].T.numpy()),   # 64   -> 16
        'b3': jnp.array(sd['layers.6.bias'].numpy()),
        'w4': jnp.array(sd['layers.7.weight'].T.numpy()),   # 16   -> 1
        'b4': jnp.array(sd['layers.7.bias'].numpy()),
    }
    print("[aligen_reward] MLP weights loaded and converted to JAX.")
    return params


def replicate_params(params, mesh):
    """Place every leaf of *params* on the replicated (fully-broadcast) sharding."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(mesh, P())

    def _put(x):
        return jax.device_put(jnp.asarray(x), rep)

    return jax.tree.map(_put, params)
