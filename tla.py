import numpy as np
import torch
import argparse
import neptune.new as neptune
import sys

sys.path.append('../')
import TD3
import utils
from hyparameters import get_hyperparameters
from common import make_env, create_folders, make_env_cc


# Main function of the policy. Model is trained and evaluated inside.
def train(seed=0, parent_steps=2, env_name='InvertedPendulum-v2', lr=3e-4, p=1, j=1, pre_gate=False, gate_replay_buffer=1000000):
    hy = get_hyperparameters(env_name)
    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    default_timestep = hy['timestep']
    default_frame_skip = hy['frame_skip']
    env_type = hy['type']
    child_response_rate = default_timestep * default_frame_skip
    print('Setting fast response rate to:', child_response_rate)

    augment_type = "gated_repetition_ma"
    arguments = [augment_type, env_name, seed, parent_steps, lr, p, j, pre_gate, gate_replay_buffer]
    file_name = '_'.join([str(x) for x in arguments])

    parameters = {
        'type': augment_type,
        'env_name': env_name,
        'seed': seed,
        'parent_steps': parent_steps,
        'child_response_rate': child_response_rate,
        'lr': lr,
        'p': p,
        'j': j,
        'pre_gate': pre_gate,
        'gate_replay_buffer': gate_replay_buffer
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    if env_type == 'mujoco':
        timestep = default_timestep if default_timestep <= child_response_rate else child_response_rate
        frame_skip = child_response_rate / timestep
        # The ratio of the default time consumption between two states returned and reset version.
        # Used to reset the max episode number to guarantee the actual max time is always the same.
        time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
        print('timestep:', timestep)  # How long does it take before two frames
        # How many frames to skip before return the state, 1 by default
        print('frameskip:', frame_skip)
        env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, False)
    else:
        timestep = child_response_rate
        time_change_factor = default_timestep / timestep
        env = make_env_cc(env_name, seed, timestep)
        print('timestep:', timestep)  # How long does it take before two frames

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    max_timesteps = hy['max_timesteps']
    eval_freq = hy['eval_freq']
    start_timesteps = hy['start_timesteps']

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # parent_expl_noise = expl_noise * ((default_timestep * default_frame_skip)/(parent_steps * timestep))

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": hy['discount'],
              "tau": hy['tau'], "observation_space": env.observation_space, "lr": lr,
              "policy_noise": hy['policy_noise'] * max_action, "noise_clip": hy['noise_clip'] * max_action,
              "policy_freq": hy['policy_freq'], "neurons": [400, 300]}

    # Target policy smoothing is scaled wrt the action scale
    if pre_gate:
        parent_policy = TD3.TempoRLTLAPreGate(**kwargs)
    else:
        parent_policy = TD3.TempoRLTLA(**kwargs)
    kwargs["state_dim"] = state_dim
    kwargs["action_dim"] = action_dim
    kwargs["delayed_env"] = False
    kwargs["reflex"] = False
    policy = TD3.TD3(**kwargs)

    replay_buffer_parent = utils.ReplayBuffer(state_dim, action_dim, max_size=hy['replay_size'])
    if pre_gate:
        skip_replay_buffer = utils.ReplayBuffer(state_dim, 1, max_size=gate_replay_buffer)
    else:
        skip_replay_buffer = utils.FiGARReplayBuffer(state_dim, action_dim, 1, max_size=hy['replay_size'])
    replay_buffer_child = utils.ReplayBuffer(state_dim, action_dim, max_size=hy['replay_size'])

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = env._max_episode_steps

    best_performance = -10000
    best_efficiency = -10000
    parent_reward = 0
    gate_reward = 0
    evaluations = []
    evaluation_decisions = []
    evaluations_fast = []
    evaluations_slow = []
    fast_actions = 0

    for t in range(int(max_timesteps)):
        # Select action randomly or according to policy
        if episode_timesteps % parent_steps == 0:
            if episode_timesteps != 0:
                # if skip == 0: # Uncomment this for multiple actions
                replay_buffer_parent.add(parent_state, parent_action, state, parent_reward, done_bool)
                if pre_gate:
                    skip_replay_buffer.add(parent_state, skip, state, gate_reward, done_bool)
                else:
                    skip_replay_buffer.add(parent_state, skip_pa, skip, state, skip_pa, gate_reward, done_bool)
            parent_state = state
            if t < start_timesteps:
                parent_action = env.action_space.sample()
                skip = np.random.randint(2)
            else:
                parent_action = (parent_policy.select_action(parent_state) + np.random.normal(0, max_action * hy[
                    'expl_noise'], size=action_dim)).clip(-max_action, max_action)
                skip = parent_policy.select_skip(parent_state, parent_action)
                if np.random.random() < hy['expl_noise']:
                    skip = np.random.randint(2)  # + 1 sonce randint samples from [0, max_rep)
                else:
                    skip = np.argmax(skip)  # + 1 since indices start at 0

            skip_pa = parent_action
            if skip>0:
                parent_reward = -p * parent_steps
                gate_reward = -p * parent_steps
            else:
                parent_reward = 0
                gate_reward = 0

        if skip > 0:
            fast_actions += 1
            if t < start_timesteps:
                child_action = env.action_space.sample()
            else:
                child_action = (policy.select_action(state) + np.random.normal(0, max_action * hy['expl_noise'],
                                                                                     size=action_dim)).clip(-max_action,
                                                                                                            max_action)
            action = child_action
        else:
            action = parent_action

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_timesteps += 1

        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        if skip > 0:
            child_reward = reward - (j * (np.mean(np.abs(action - parent_action)) / max_action))
            replay_buffer_child.add(state, action, next_state, child_reward, done_bool)
            parent_reward += reward - (j * (np.mean(np.abs(action - parent_action)) / max_action))
            gate_reward += reward - (j * (np.mean(np.abs(action - parent_action)) / max_action))
        else:
            replay_buffer_child.add(state, action, next_state, reward, done_bool)
            parent_reward += reward
            gate_reward += reward

        state = next_state

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # if skip == 0:
            replay_buffer_parent.add(parent_state, parent_action, state, parent_reward, done_bool)
            if pre_gate:
                skip_replay_buffer.add(parent_state, skip, state, gate_reward, done_bool)
            else:
                skip_replay_buffer.add(parent_state, skip_pa, skip, state, skip_pa, gate_reward, done_bool)
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Fast Actions:{fast_actions:.3f} Reward: {episode_reward:.3f}")
            # Reset environment
            fast_actions = 0
            state, done = env.reset(), False
            episode_reward = 0
            episode_num += 1
            parent_reward = 0
            gate_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            if env_type == 'mujoco':
                eval_env = make_env(env_name, seed + 100, time_change_factor, timestep, frame_skip, False)
            else:
                eval_env = make_env_cc(env_name, seed + 100, timestep)
            task_reward = 0
            eval_decisions = 0
            slow_actions = 0
            fast_decisions = 0
            for _ in range(10):
                eval_state, eval_done = eval_env.reset(), False
                eval_episode_timesteps = 0
                while not eval_done:
                    if eval_episode_timesteps % parent_steps == 0:
                        eval_parent_action = parent_policy.select_action(eval_state)
                        eval_skip = np.argmax(parent_policy.select_skip(eval_state, eval_parent_action))
                        eval_action = eval_parent_action
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
            avg_fast_actions = fast_decisions/10
            evaluations.append(avg_reward)
            evaluation_decisions.append(avg_decisions)
            evaluations_slow.append(avg_slow_actions)
            evaluations_fast.append(avg_fast_actions)
            print(
                f" --------------- Slow Actions {avg_slow_actions:.3f}, Decisions {avg_decisions:.3f}, Evaluation reward {avg_reward:.3f}")
            run['avg_reward'].log(avg_reward)
            run['avg_decisions'].log(avg_decisions)
            run['avg_slow'].log(avg_slow_actions)
            run['avg_fast'].log(avg_fast_actions)

            np.save(f"./results/{file_name}", evaluations)
            np.save(f"./results/{file_name}_decisions", evaluation_decisions)
            np.save(f"./results/{file_name}_slow_decisions", evaluations_slow)
            np.save(f"./results/{file_name}_fast_decisions", evaluations_fast)


            if best_efficiency <= avg_reward / avg_decisions:
                best_efficiency = avg_reward / avg_decisions
                run['best_efficiency'].log(best_efficiency)
                parent_policy.save(f"./models/{file_name}_best_efficiency")
                policy.save(f"./models/{file_name}_child_best_efficiency")

            if best_performance <= avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance)
                parent_policy.save(f"./models/{file_name}_best")  # if t % parent_steps == 0:
                policy.save(f"./models/{file_name}_child_best")

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            parent_policy.train(replay_buffer_parent, hy['batch_size'])
            parent_policy.train_skip(skip_replay_buffer, hy['batch_size'])
            policy.train(replay_buffer_child, hy['batch_size'])

    parent_policy.save(f"./models/{file_name}_final")
    policy.save(f"./models/{file_name}_child_final")

    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--parent_steps", default=2, type=int, help="Number of steps by parent")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--p", default=1.0, type=float, help="reward penalty for the slow network")
    parser.add_argument("--j", default=1.0, type=float, help="reward penalty for the fast network")
    parser.add_argument("--pre_gate", action="store_true", help="Gate before lazy network")
    parser.add_argument("--gate_replay_buffer", default=1000000, type=int, help="gate replay buffer")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
