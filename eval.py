import argparse
import numpy as np
import torch
import model
import tqdm
from common import make_env, load_config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def eval(env_name):
    config = load_config("configs/config.yaml", env_name)
    env = make_env(env_name, 0)
    # Initialize models
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": config["discount"],
        "tau": config["tau"],
        "lr": config["lr"],
        "policy_noise": config["policy_noise"] * max_action,
        "noise_clip": config["noise_clip"] * max_action,
        "policy_freq": config["policy_freq"],
        "neurons": [400, 300],
    }
    slow_steps = config["slow_steps"]
    rewards = []
    repetitions = []
    actions = []
    decisions = []
    jerks = []
    slow_actions = []
    fast_actions = []

    for seed in tqdm.tqdm_gui(range(10)):
        slow_policy = model.TLA(**kwargs)
        policy = model.TD3(**kwargs)
        arguments = [
            "TLA",
            env_name,
            seed,
            slow_steps,
            config["lr"],
            config["p"],
            config["j"],
        ]
        file_name = "_".join([str(x) for x in arguments])
        slow_policy.load(f"./models/{file_name}_best")
        policy.load(f"./models/{file_name}_fast_best")

        set_seed(seed+100)
        eval_env = make_env(env_name, 100 + seed)
        action_reptition = 0
        total_actions = 0
        total_decisions = 0
        task_reward = 0
        slow_action = 0
        fast_action = 0
        jerk = 0
        for _ in range(10):
            eval_state, eval_done = eval_env.reset()
            eval_episode_timesteps = 0
            prev_action = -100

            while not eval_done:
                if eval_episode_timesteps % slow_steps == 0:
                    eval_parent_action = slow_policy.select_action(eval_state)
                    eval_skip = np.argmax(slow_policy.select_skip(eval_state, eval_parent_action))
                    eval_action = eval_parent_action
                    slow_action += 1
                    if eval_skip == 0:
                        total_decisions += 1
                if eval_skip > 0:
                    fast_action += 1
                    total_decisions += 1
                    eval_action = policy.select_action(eval_state)

                eval_next_state, eval_reward, eval_d,tr, _ = eval_env.step(eval_action)
                eval_state = eval_next_state
                eval_done=(eval_d or tr)
                eval_episode_timesteps += 1
                if prev_action == eval_action:
                    action_reptition += 1
                jerk += abs(eval_action - prev_action)
                prev_action = eval_action
                total_actions += 1
                task_reward += eval_reward

                # if t > 25000:
                #     eval_env.render()
        rewards.append(task_reward / 10)
        repetitions.append(action_reptition / total_actions)
        actions.append(total_actions / 10)
        decisions.append(total_decisions / 10)
        jerks.append(jerk / total_actions)
        slow_actions.append(slow_action / 10)
        fast_actions.append(fast_action / 10)

    print(f'--------------{env_name}----------------')
    print(f"Mean Rewards: {np.mean(rewards)}")
    print(f"STD Rewards: {np.std(rewards)}")
    print(f"mean Repetitions: {np.mean(repetitions)}")
    print(f"Mean actions: {np.mean(actions)}")
    print(f"Mean decisions: {np.mean(decisions)}")
    print(f"Mean jerk: {np.mean(jerks)}")
    print(f"Repetitions%: {np.mean(repetitions)* 100}%")
    print(f"Mean slow actions: {np.mean(slow_actions)}")
    print(f"Mean fast actions: {np.mean(fast_actions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for TLA model.")
    parser.add_argument("--env_name", default="Pendulum-v1", help="Environment name")
    args = parser.parse_args()

    eval(env_name=args.env_name)
