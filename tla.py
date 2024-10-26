import numpy as np
import torch
import argparse
import neptune.new as neptune
import sys
import logging
import time
from typing import Dict, Any

import TD3
import utils
from hyparameters import get_hyperparameters
from common import make_env, create_folders, make_env_cc

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the default level to INFO
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"training_tla_{int(time.time())}.log"),  # Logs to a file named 'training.log'
        logging.StreamHandler()  # Also output logs to the console
    ]
)


def initialize_neptune_run(project_name: str) -> Any:
    """
    Initialize a Neptune run.

    Parameters
    ----------
    project_name : str
        The name of the Neptune project.

    Returns
    -------
    Neptune run object
    """
    run = neptune.init(
        project=project_name,
        api_token="YOUR_API_TOKEN_HERE"  # Replace with environment variable for security
    )
    return run


def setup_environment(env_name: str, seed: int) -> Any:
    """
    Set up the environment based on the type (e.g., mujoco).

    Parameters
    ----------
    env_name : str
        Name of the environment.
    seed : int
        Seed for reproducibility.
    Returns
    -------
    Environment object
    """
    env = make_env(env_name, seed)

    logging.info(f"Environment setup complete. Env: {env_name}, Seed: {seed}")
    return env


def setup_policies(state_dim: int, action_dim: int, max_action: float, hy: Dict[str, Any], lr: float,
                   pre_gate: bool) -> Any:
    """
    Set up the policies for training.

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space.
    max_action : float
        Maximum action value.
    hy : dict
        Hyperparameters for the environment.
    lr : float
        Learning rate.
    pre_gate : bool
        Use gate before lazy network.

    Returns
    -------
    Tuple containing the parent policy and child policy
    """
    kwargs = {
        "state_dim": state_dim, "action_dim": action_dim, "max_action": max_action,
        "discount": hy['discount'], "tau": hy['tau'], "observation_space": hy['observation_space'],
        "lr": lr, "policy_noise": hy['policy_noise'] * max_action,
        "noise_clip": hy['noise_clip'] * max_action, "policy_freq": hy['policy_freq'], "neurons": [400, 300]
    }
    parent_policy = TD3.TempoRLTLAPreGate(**kwargs) if pre_gate else TD3.TempoRLTLA(**kwargs)
    policy = TD3.TD3(**kwargs)

    logging.info("Policies setup complete.")
    return parent_policy, policy


def setup_replay_buffers(state_dim: int, action_dim: int, hy: Dict[str, Any], pre_gate: bool,
                         gate_replay_buffer: int) -> Any:
    """
    Set up the replay buffers for training.

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space.
    hy : dict
        Hyperparameters for the environment.
    pre_gate : bool
        Use gate before lazy network.
    gate_replay_buffer : int
        Size of the gate replay buffer.

    Returns
    -------
    Tuple containing the replay buffers for parent, child, and skip
    """
    replay_buffer_parent = utils.ReplayBuffer(state_dim, action_dim, max_size=hy['replay_size'])
    skip_replay_buffer = utils.ReplayBuffer(state_dim, 1, max_size=gate_replay_buffer) if pre_gate else \
        utils.FiGARReplayBuffer(state_dim, action_dim, 1, max_size=hy['replay_size'])
    replay_buffer_child = utils.ReplayBuffer(state_dim, action_dim, max_size=hy['replay_size'])

    logging.info("Replay buffers setup complete.")
    return replay_buffer_parent, replay_buffer_child, skip_replay_buffer


def training_loop(env, parent_policy, policy, replay_buffer_parent, replay_buffer_child, skip_replay_buffer, hy,
                  max_action, parent_steps, start_timesteps):
    """
    Execute the main training loop.

    Parameters
    ----------
    env : Environment
        The environment for training.
    parent_policy : Policy
        The parent policy used for training.
    policy : Policy
        The child policy used for training.
    replay_buffer_parent : ReplayBuffer
        Replay buffer for parent policy.
    replay_buffer_child : ReplayBuffer
        Replay buffer for child policy.
    skip_replay_buffer : ReplayBuffer
        Replay buffer for skip decisions.
    hy : dict
        Hyperparameters for the environment.
    max_action : float
        Maximum action value.
    parent_steps : int
        Number of steps by parent policy.
    start_timesteps : int
        Number of timesteps before training starts.
    """
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(hy['max_timesteps'])):
        action = select_action(t, start_timesteps, env, state, parent_policy, max_action, hy)
        next_state, reward, done, done_bool = perform_action(env, state, action, episode_timesteps)
        store_transition(replay_buffer_parent, state, action, next_state, reward, done_bool)

        state, episode_reward, episode_timesteps, episode_num = update_state(env, state, next_state, reward, done,
                                                                             episode_reward, episode_timesteps,
                                                                             episode_num)

        if t >= start_timesteps:
            train_policies(parent_policy, policy, replay_buffer_parent, replay_buffer_child, hy)

        if done:
            logging.info(
                f"Episode {episode_num} finished. Total Timesteps: {episode_timesteps}, Reward: {episode_reward:.2f}")

    logging.info("Training loop complete.")


def select_action(t: int, start_timesteps: int, env, state, parent_policy, max_action: float, hy: Dict[str, Any]):
    """
    Select an action based on the current policy or randomly.

    Parameters
    ----------
    t : int
        Current timestep.
    start_timesteps : int
        Number of timesteps before training starts.
    env : Environment
        The environment.
    state : np.array
        Current state.
    parent_policy : Policy
        The parent policy.
    max_action : float
        Maximum action value.
    hy : dict
        Hyperparameters.

    Returns
    -------
    Action to be taken
    """
    if t < start_timesteps:
        action = env.action_space.sample()
        logging.debug(f"Timestep {t}: Random action selected.")
    else:
        action = (parent_policy.select_action(np.array(state)) +
                  np.random.normal(0, max_action * hy['expl_noise'], size=env.action_space.shape[0])).clip(-max_action,
                                                                                                           max_action)
        logging.debug(f"Timestep {t}: Action selected by policy.")
    return action


def perform_action(env, state, action, episode_timesteps: int):
    """
    Perform the action in the environment.

    Parameters
    ----------
    env : Environment
        The environment.
    state : np.array
        Current state.
    action : np.array
        Action to be performed.
    episode_timesteps : int
        Current episode timestep count.

    Returns
    -------
    next_state, reward, done, done_bool
    """
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps + 1 < env._max_episode_steps else 0
    logging.debug(f"Action performed. Reward: {reward}, Done: {done}")
    return next_state, reward, done, done_bool


def store_transition(replay_buffer, state, action, next_state, reward, done_bool):
    """
    Store the transition in the replay buffer.

    Parameters
    ----------
    replay_buffer : ReplayBuffer
        Replay buffer to store the transition.
    state : np.array
        Current state.
    action : np.array
        Action taken.
    next_state : np.array
        Next state after taking the action.
    reward : float
        Reward received.
    done_bool : float
        Whether the episode is done.
    """
    replay_buffer.add(state, action, next_state, reward, done_bool)
    logging.debug("Transition stored in replay buffer.")


def update_state(env, state, next_state, reward, done, episode_reward, episode_timesteps, episode_num):
    """
    Update the state and episode information.

    Parameters
    ----------
    env : Environment
        The environment.
    state : np.array
        Current state.
    next_state : np.array
        Next state after action.
    reward : float
        Reward received.
    done : bool
        Whether the episode is done.
    episode_reward : float
        Accumulated reward for the episode.
    episode_timesteps : int
        Current episode timestep count.
    episode_num : int
        Current episode number.

    Returns
    -------
    Updated state, episode_reward, episode_timesteps, episode_num
    """
    episode_reward += reward
    episode_timesteps += 1

    if done:
        logging.info(f"Episode {episode_num} completed with reward: {episode_reward}")
        state, done = env.reset(), False
        episode_timesteps = 0
        episode_num += 1
    else:
        state = next_state

    return state, episode_reward, episode_timesteps, episode_num


def train_policies(parent_policy, policy, replay_buffer_parent, replay_buffer_child, hy: Dict[str, Any]):
    """
    Train the policies based on the collected data.

    Parameters
    ----------
    parent_policy : Policy
        The parent policy to be trained.
    policy : Policy
        The child policy to be trained.
    replay_buffer_parent : ReplayBuffer
        Replay buffer for parent policy.
    replay_buffer_child : ReplayBuffer
        Replay buffer for child policy.
    hy : dict
        Hyperparameters for training.
    """
    parent_policy.train(replay_buffer_parent, hy['batch_size'])
    policy.train(replay_buffer_child, hy['batch_size'])
    logging.debug("Policies trained.")


def train(seed: int = 0, parent_steps: int = 2, env_name: str = 'InvertedPendulum-v2', lr: float = 3e-4, p: float = 1.0,
          j: float = 1.0, pre_gate: bool = False, gate_replay_buffer: int = 1000000) -> None:
    """
    Main training function for the policy.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    parent_steps : int
        Number of steps by parent policy.
    env_name : str
        Environment name.
    lr : float
        Learning rate.
    p : float
        Reward penalty for the slow network.
    j : float
        Reward penalty for the fast network.
    pre_gate : bool
        Use gate before lazy network.
    gate_replay_buffer : int
        Size of the gate replay buffer.
    """
    # Load hyperparameters and initialize Neptune
    hy = get_hyperparameters(env_name)
    run = initialize_neptune_run("dee0512/Reflex")

    # Environment setup
    env = setup_environment(env_name, seed)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    logging.info("Random seeds set for reproducibility.")

    # Extract environment details
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Policy and replay buffer setup
    parent_policy, policy = setup_policies(env, hy, lr, pre_gate)
    replay_buffer_parent, replay_buffer_child, skip_replay_buffer = setup_replay_buffers(state_dim, action_dim, hy,
                                                                                         pre_gate, gate_replay_buffer)

    # Training loop
    training_loop(env, parent_policy, policy, replay_buffer_parent, replay_buffer_child, skip_replay_buffer, hy,
                  max_action, parent_steps, hy['start_timesteps'])

    logging.info("Training complete.")
    run.stop()
    logging.info("Neptune run stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--parent_steps", default=2, type=int, help="Number of steps by parent")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--p", default=1.0, type=float, help="Reward penalty for the slow network")
    parser.add_argument("--j", default=1.0, type=float, help="Reward penalty for the fast network")
    parser.add_argument("--pre_gate", action="store_true", help="Gate before lazy network")
    parser.add_argument("--gate_replay_buffer", default=1000000, type=int, help="Gate replay buffer size")

    args = parser.parse_args()
    args = vars(args)

    train(**args)
