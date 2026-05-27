"""FAV fine-tuning. RBF-kernel SVGD toward p(x)*r(x).

Loss: MSE(gen, gen + stein_velocity), where
      stein_velocity = K @ (prior_score + beta * nabla_log_r) - grad_K  (driving + repulsive).

For MeanFlowModel: pass sample_kwargs={"n_steps": N} to backprop through N ODE steps.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Callable, Dict, Optional

from utils_2d import (
    BaseGenerator,
    evaluate_metrics_finetune,
    save_metrics_csv,
)

def fav_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    energy_fn: Callable,
    temp: float,
    beta: float = 1.0,
) -> torch.Tensor:
    n = gen.size(0)
    gen_req = gen.detach().clone().requires_grad_(True)

    r_out = energy_fn(gen_req)
    log_r = torch.log(r_out.clamp_min(1e-10)).sum()
    nabla_log_r = torch.autograd.grad(log_r, gen_req, create_graph=True)[0]

    # KDE-based prior score nabla log p(x) from `pos` samples.
    dist_pos = torch.cdist(gen_req, pos.detach(), p=2) ** 2
    kernel_pos = (-dist_pos / temp).exp()
    Z_p = kernel_pos.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prior_score = 2.0 * (kernel_pos @ pos - Z_p * gen_req) / (temp * Z_p)

    dist_gen = torch.cdist(gen_req, gen_req, p=2) ** 2
    kernel_matrix = (-dist_gen / temp).exp()

    total_score = prior_score + beta * nabla_log_r
    driving_term = (kernel_matrix.detach() @ total_score) / n
    grad_k = torch.autograd.grad(kernel_matrix.sum(), gen_req)[0]
    repulsive_term = -grad_k / (2 * n)

    stein_velocity = driving_term + repulsive_term
    return F.mse_loss(gen, (gen + stein_velocity).detach())

def finetune(
    model: BaseGenerator,
    data: torch.Tensor,
    device: torch.device,
    energy_fn: Callable,
    *,
    num_steps: int = 20000,
    batch_size: int = 2048,
    lr: float = 1e-4,
    beta: float = 1.0,
    temp: float = 0.05,
    metric_every: int = 200,
    metric_samples: int = 4096,
    save_dir: str = "results/finetune/fav",
    dataset_name: str = "data",
    ref_data: Optional[torch.Tensor] = None,
    sample_kwargs: Optional[Dict] = None,
    pos_pool: torch.Tensor = None,
) -> BaseGenerator:
    if sample_kwargs is None:
        sample_kwargs = {}

    # KDE prior is always sampled from the frozen pretrained model.
    assert pos_pool is not None, "pos_pool (frozen-pretrained samples) is required."
    pos_source = pos_pool
    print(f"[FAV] pos source: frozen pretrained pool, N={pos_source.shape[0]:,}")

    os.makedirs(save_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps_tag = f"  n_steps={sample_kwargs['n_steps']}" if "n_steps" in sample_kwargs else ""
    print(f"[FAV] bandwidth temp={temp:.4f}  beta={beta:.3f}{n_steps_tag}")

    losses, m_steps, kl_list, mmd_list, reward_list = [], [], [], [], []
    ema = None
    last_kl, last_mmd2, last_reward = None, None, None
    EMA_ALPHA = 0.3
    ema_mmd2 = None
    best_ema_mmd2 = float("inf")
    best_state = None
    best_step = -1

    pbar = tqdm(range(num_steps), desc=f"FAV{n_steps_tag}")

    for step in pbar:
        idx = torch.randint(0, pos_source.shape[0], (batch_size,), device=device)
        pos = pos_source[idx]

        # Gradients must flow through ODE steps for MeanFlow.
        gen = model.sample(batch_size, device, **sample_kwargs)

        loss = fav_loss(gen, pos, energy_fn, temp, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()

        losses.append(loss.item())
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.3e}")

        if step % metric_every == 0:
            pool = ref_data if ref_data is not None else data[:10000]
            kl, mmd2, reward_mean = evaluate_metrics_finetune(
                model, pool, energy_fn, device, n=metric_samples, beta=beta, **sample_kwargs
            )
            last_kl, last_mmd2, last_reward = kl, mmd2, reward_mean
            m_steps.append(step); kl_list.append(kl); mmd_list.append(mmd2); reward_list.append(reward_mean)
            ema_mmd2 = mmd2 if ema_mmd2 is None else EMA_ALPHA * mmd2 + (1 - EMA_ALPHA) * ema_mmd2
            if ema_mmd2 < best_ema_mmd2:
                best_ema_mmd2 = ema_mmd2
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            pbar.set_postfix(loss=f"{ema:.3e}", KL=f"{kl:.3f}", MMD2=f"{mmd2:.4f}",
                             r=f"{reward_mean:.3f}", bestEMA=f"{best_ema_mmd2:.4f}")

    save_metrics_csv(os.path.join(save_dir, "metrics_finetune.csv"), m_steps, kl_list, mmd_list,
                     reward_vals=reward_list)

    # Returned model has LAST weights; best state is saved separately.
    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, "model_best.pt"))
        print(f"[FAV] best EMA-MMD2 = {best_ema_mmd2:.6e} @ step {best_step}")
    return model
