import logging
import numpy as np
import torch
import model
import utils
from common import make_env, create_folders
import yaml
from datetime import datetime

logger = None


# Configure logging
def configure_logging(env_name, seed):
    logs_dir = "./logs"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{logs_dir}/training_{env_name}_seed{seed}_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)



def load_config(config_path, env_name):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if "default" not in config or "environments" not in config:
        raise ValueError("Config file must have 'default' and 'environments' sections.")

    # Load default and environment-specific configurations
    default_config = config["default"]
    env_config = config["environments"].get(env_name, {})

    if env_config is None:
        env_config = {}

    # Merge configurations (environment-specific settings take precedence)
    final_config = {**default_config, **env_config}
    return final_config


def train(run, seed=0, env_name="InvertedPendulum-v2"):
    create_folders()
    global logger
    logger = configure_logging(env_name=env_name, seed=seed)

    # Load configuration for the specified environment
    config = load_config("configs/config.yaml", env_name)
    arguments = [
        "TLA",
        env_name,
        seed,
        config["slow_steps"],
        config["lr"],
        config["p"],
        config["j"],
    ]
    file_name = "_".join([str(x) for x in arguments])

    parameters = {
        "type": "TLA",
        "env_name": env_name,
        "seed": seed,
        "slow_steps": config["slow_steps"],
        "lr": config["lr"],
        "p": config["p"],
        "j": config["j"],
    }
    run["parameters"] = parameters
    logger.info(f"Starting training with Env: {env_name}, Seed: {seed}")

    env, slow_policy, policy, replay_buffers = initialize_training(
        env_name,
        seed,
        config["lr"],
        config["discount"],
        config["tau"],
        config["policy_noise"],
        config["noise_clip"],
        config["policy_freq"],
        config["replay_size"],
    )

    # Training Loop
    train_loop(
        run,
        env,
        slow_policy,
        policy,
        replay_buffers,
        file_name,
        config["slow_steps"],
        config["max_timesteps"],
        config["eval_freq"],
        config["start_timesteps"],
        config["p"],
        config["j"],
        config["expl_noise"],
        env_name,
        seed,
        config["batch_size"],
    )


def initialize_training(
    env_name,
    seed,
    lr,
    discount,
    tau,
    policy_noise,
    noise_clip,
    policy_freq,
    replay_size,
):
    # Initialize environment
    env = make_env(env_name, seed)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize models
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "lr": lr,
        "policy_noise": policy_noise * max_action,
        "noise_clip": noise_clip * max_action,
        "policy_freq": policy_freq,
        "neurons": [400, 300],
    }

    slow_policy = model.TLA(**kwargs)
    policy = model.TD3(**kwargs)

    # Initialize replay buffers
    slow_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=replay_size)
    skip_replay_buffer = utils.FiGARReplayBuffer(
        state_dim, action_dim, 1, max_size=replay_size
    )
    fast_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=replay_size)
    logger.debug("Initialized environment, models, and replay buffers.")
    return (
        env,
        slow_policy,
        policy,
        (slow_replay_buffer, skip_replay_buffer, fast_replay_buffer),
    )


def train_loop(
    run,
    env,
    slow_policy,
    policy,
    replay_buffers,
    file_name,
    slow_steps,
    max_timesteps,
    eval_freq,
    start_timesteps,
    p,
    j,
    expl_noise,
    env_name,
    seed,
    batch_size,
):
    slow_replay_buffer, skip_replay_buffer, fast_replay_buffer = replay_buffers
    state, done = env.reset(), False
    slow_state = state
    skip = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = env._max_episode_steps
    best_performance = -10000
    best_efficiency = -10000
    slow_reward = 0
    gate_reward = 0
    evaluations = []
    evaluation_decisions = []
    evaluations_fast = []
    evaluations_slow = []
    fast_actions = 0
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    slow_action = np.random.normal(0, max_action * expl_noise, size=action_dim)
    for t in range(int(max_timesteps)):
        # Select action
        if episode_timesteps % slow_steps == 0:
            (
                slow_state,
                slow_action,
                skip,
                slow_reward,
                gate_reward,
            ) = select_slow_action(
                env,
                slow_policy,
                state,
                t,
                start_timesteps,
                expl_noise,
                slow_steps,
                p,
                slow_reward,
                gate_reward,
                episode_timesteps,
                slow_replay_buffer,
                skip_replay_buffer,
                slow_state,
                slow_action,
                skip,
                max_action,
                action_dim,
            )

        if skip > 0:
            fast_actions += 1
            if t < start_timesteps:
                fast_action = env.action_space.sample()
            else:
                fast_action = (
                    policy.select_action(state)
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            action = fast_action
        else:
            action = slow_action

        # Environment step
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_timesteps += 1
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        update_fast_replay_buffer(
            fast_replay_buffer,
            state,
            slow_action,
            action,
            next_state,
            reward,
            done_bool,
            skip,
            j,
            max_action,
        )

        if skip > 0:
            reward_penalty = j * (np.mean(np.abs(action - slow_action)) / max_action)
            slow_reward += reward - reward_penalty
            gate_reward += reward - reward_penalty
        else:
            slow_reward += reward
            gate_reward += reward

        state = next_state

        if done:
            handle_episode_end(
                replay_buffers,
                slow_state,
                slow_action,
                skip,
                state,
                slow_reward,
                gate_reward,
                episode_num,
                episode_reward,
                done_bool,
                t,
            )

            state, done = env.reset(), False
            episode_reward = 0
            episode_num += 1
            slow_reward = 0
            gate_reward = 0
            episode_timesteps = 0
            fast_actions = 0

        # Evaluate and save models if necessary
        if (t + 1) % eval_freq == 0:
            (
                avg_reward,
                avg_decisions,
                avg_slow_actions,
                avg_fast_actions,
            ) = evaluate_policy(env_name, seed, slow_steps, slow_policy, policy)
            evaluations.append(avg_reward)
            evaluation_decisions.append(avg_decisions)
            evaluations_slow.append(avg_slow_actions)
            evaluations_fast.append(avg_fast_actions)
            logger.info(
                f" Evaluation at step {t + 1}: Avg Reward: {avg_reward:.3f}, Avg Decisions: {avg_decisions:.3f}, Avg Slow Actions: {avg_slow_actions:.3f}"
            )
            run["avg_reward"].log(avg_reward)
            run["avg_decisions"].log(avg_decisions)
            run["avg_slow"].log(avg_slow_actions)
            run["avg_fast"].log(avg_fast_actions)
            save_model(
                avg_reward,
                avg_decisions,
                best_efficiency,
                best_performance,
                run,
                slow_policy,
                policy,
                file_name,
            )

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            slow_policy.train(slow_replay_buffer, batch_size)
            slow_policy.train_skip(skip_replay_buffer, batch_size)
            policy.train(fast_replay_buffer, batch_size)

        logger.debug("Training loop completed.")


def select_slow_action(
    env,
    slow_policy,
    state,
    t,
    start_timesteps,
    expl_noise,
    slow_steps,
    energy_penalty,
    slow_reward,
    gate_reward,
    episode_timesteps,
    slow_replay_buffer,
    skip_replay_buffer,
    slow_state,
    slow_action,
    skip,
    max_action,
    action_dim,
):
    if episode_timesteps != 0:
        slow_replay_buffer.add(slow_state, slow_action, state, slow_reward, 0)
        skip_replay_buffer.add(
            slow_state, slow_action, skip, state, slow_action, gate_reward, 0
        )

    slow_state = state
    if t < start_timesteps:
        slow_action = env.action_space.sample()
        skip = np.random.randint(2)
    else:
        slow_action = (
            slow_policy.select_action(slow_state)
            + np.random.normal(0, max_action * expl_noise, size=action_dim)
        ).clip(-max_action, max_action)
        skip = slow_policy.select_skip(slow_state, slow_action)
        if np.random.random() < expl_noise:
            skip = np.random.randint(2)
        else:
            skip = np.argmax(skip)

    if skip > 0:
        slow_reward = -energy_penalty * slow_steps
        gate_reward = -energy_penalty * slow_steps
    else:
        slow_reward = 0
        gate_reward = 0

    return slow_state, slow_action, skip, slow_reward, gate_reward


def update_fast_replay_buffer(
    replay_buffer,
    state,
    slow_action,
    action,
    next_state,
    reward,
    done_bool,
    skip,
    j,
    max_action,
):
    if skip > 0:
        fast_reward = reward - (
            j * (np.mean(np.abs(action - slow_action)) / max_action)
        )
        replay_buffer.add(state, action, next_state, fast_reward, done_bool)
    else:
        replay_buffer.add(state, action, next_state, reward, done_bool)


def handle_episode_end(
    replay_buffers,
    slow_state,
    slow_action,
    skip,
    state,
    slow_reward,
    gate_reward,
    episode_num,
    episode_reward,
    done_bool,
    t,
):
    slow_replay_buffer, skip_replay_buffer, fast_replay_buffer = replay_buffers
    slow_replay_buffer.add(slow_state, slow_action, state, slow_reward, done_bool)
    skip_replay_buffer.add(
        slow_state, slow_action, skip, state, slow_action, gate_reward, done_bool
    )
    logger.info(
        f"Total T: {t + 1} Episode Num: {episode_num + 1} Reward: {episode_reward:.3f}"
    )


def evaluate_policy(env_name, seed, slow_steps, slow_policy, policy):
    eval_env = make_env(env_name, seed + 100)
    task_reward = 0
    eval_decisions = 0
    slow_actions = 0
    fast_decisions = 0
    for _ in range(10):
        eval_state, eval_done = eval_env.reset(), False
        eval_episode_timesteps = 0
        while not eval_done:
            if eval_episode_timesteps % slow_steps == 0:
                eval_slow_action = slow_policy.select_action(eval_state)
                eval_skip = np.argmax(
                    slow_policy.select_skip(eval_state, eval_slow_action)
                )
                eval_action = eval_slow_action
                slow_actions += 1
                if eval_skip == 0:
                    eval_decisions += 1
            if eval_skip > 0:
                eval_decisions += 1
                fast_decisions += 1
                eval_action = policy.select_action(eval_state)
            eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
            eval_state = eval_next_state
            eval_episode_timesteps += 1
            task_reward += eval_reward
    avg_reward = task_reward / 10
    avg_decisions = eval_decisions / 10
    avg_slow_actions = slow_actions / 10
    avg_fast_actions = fast_decisions / 10
    return avg_reward, avg_decisions, avg_slow_actions, avg_fast_actions


def save_model(
    avg_reward,
    avg_decisions,
    best_efficiency,
    best_performance,
    run,
    slow_policy,
    policy,
    file_name,
):
    if best_efficiency <= avg_reward / avg_decisions:
        best_efficiency = avg_reward / avg_decisions
        run["best_efficiency"].log(best_efficiency)
        slow_policy.save(f"./models/{file_name}_best_efficiency")
        policy.save(f"./models/{file_name}_fast_best_efficiency")
    if best_performance <= avg_reward:
        best_performance = avg_reward
        run["best_reward"].log(best_performance)
        slow_policy.save(f"./models/{file_name}_best")
        policy.save(f"./models/{file_name}_fast_best")
