import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


def get_adaptive_bandwidth(actions, method, default_bandwidth):
    """Compute the SVGD kernel bandwidth sigma = 2*h^2 ('fixed' or 'scott')."""
    if method == 'fixed':
        return {'bandwidth': default_bandwidth}
    if method != 'scott':
        raise ValueError(
            f"Unknown bandwidth_method {method!r}; expected 'fixed' or 'scott'."
        )

    n_samples, action_dim = actions.shape
    std = jnp.std(actions, axis=0).mean()
    safe_std = jnp.maximum(std, 1e-4)
    n_scale = n_samples ** (-1.0 / (action_dim + 4.0))
    h = n_scale * safe_std
    return {'bandwidth': 2.0 * (h ** 2), 'std': std}

class FAVAgent(flax.struct.PyTreeNode):
    """FAV agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def get_drift(self, pos, gen, state, bandwidth=None):
        bandwidth = bandwidth if bandwidth is not None else self.config['bandwidth']
        n = gen.shape[0]

        dist_pos = jnp.sum((gen[:, None, :] - jax.lax.stop_gradient(pos)[None, :, :]) ** 2, axis=-1)
        kernel_pos = jnp.exp(-dist_pos / bandwidth)
        Z_p = jnp.clip(kernel_pos.sum(axis=-1, keepdims=True), a_min=1e-6)

        prior_score = 2 * (kernel_pos @ pos - Z_p * gen) / (bandwidth * Z_p)

        dist_gen = jnp.sum((gen[:, None, :] - gen[None, :, :]) ** 2, axis=-1)
        kernel_matrix = jnp.exp(-dist_gen / bandwidth)

        state_repeated = jnp.repeat(state[None, ...], n, axis=0)

        def q_fn(action, state_single):
            qs = self.network.select('critic')(state_single[None, ...], actions=action[None, ...])
            q = jnp.mean(qs)
            return q

        nabla_q = jax.lax.stop_gradient(
            jax.vmap(
                jax.grad(q_fn),
                in_axes=(0, 0),
            )(gen, state_repeated)
        )

        driving_term = (jax.lax.stop_gradient(kernel_matrix) @ (prior_score + self.config['beta'] * nabla_q)) / n

        def kernel_sum_fn(gen_):
            dist_gen_ = jnp.sum((gen_[:, None, :] - gen_[None, :, :]) ** 2, axis=-1)
            kernel_matrix_ = jnp.exp(-dist_gen_ / bandwidth)
            return kernel_matrix_.sum()

        grad_k = jax.grad(kernel_sum_fn)(gen)
        repulsive_term = - grad_k / (2 * n)
        drift = driving_term + repulsive_term

        return drift, prior_score, nabla_q

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FAV critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FAV actor loss."""
        batch_size, action_dim = batch['actions'].shape
        gen_multiplier = self.config['gen_multiplier']
        rng, gen_rng = jax.random.split(rng, 2)

        obs_repeated = jnp.repeat(batch['observations'], gen_multiplier, axis=0)

        noises = jax.random.normal(
            gen_rng,
            (
                *obs_repeated.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        gen = self.network.select('actor')(obs_repeated, noises, params=grad_params)
        gen = jnp.clip(gen, -1, 1)

        gen = gen.reshape(batch_size, gen_multiplier, action_dim)
        pos = batch['actions'][:, jnp.newaxis, :] 

        bandwidth_metrics = get_adaptive_bandwidth(
            batch['actions'], 
            method=self.config.get('bandwidth_method', 'fixed'), 
            default_bandwidth=self.config.get('bandwidth', 1.0)
        )
        bandwidth = bandwidth_metrics['bandwidth']

        drift_fn = jax.vmap(lambda p, g, s: self.get_drift(p, g, s, bandwidth))
        drift, prior_score, nabla_q = drift_fn(pos, gen, batch['observations'])

        target_action = jax.lax.stop_gradient(gen + drift)
        drift_loss = jnp.mean(jnp.square(target_action - gen))

        prior_score_norm = jnp.linalg.norm(prior_score, axis=-1).mean()
        nabla_q_norm = jnp.linalg.norm(nabla_q, axis=-1).mean()
        gen_abs_mean = jnp.abs(gen).mean()

        info = {
            'drift_norm': jnp.linalg.norm(drift, axis=-1).mean(),
            'prior_score_norm': prior_score_norm,
            'nabla_q_norm': nabla_q_norm,
            'gen_abs_mean': gen_abs_mean,
            'bandwidth': bandwidth, # Add bandwidth variable to metrics
        }
        
        for k, v in bandwidth_metrics.items():
            if k != 'bandwidth':
                info[f'bandwidth_{k}'] = v
                
        return drift_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor') is not None:
            # Add actor_encoder to ModuleDict to make it separately callable.
            network_info['actor_encoder'] = (encoders.get('actor'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fav',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            beta=10.0,  # Temperature parameter beta (reward-guidance strength; tune per environment).
            bandwidth=1.0,  # Kernel bandwidth sigma (used when bandwidth_method == 'fixed').
            bandwidth_method='fixed',  # Bandwidth rule: 'fixed' or 'scott'.
            gen_multiplier=8,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
