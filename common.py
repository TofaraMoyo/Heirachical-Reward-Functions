import gym
import torch
import os
import yaml

__all__ = ["make_env", "create_folders", "load_config"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecisionWrapper(gym.Wrapper):
    def __init__(self, env, decisions):
        super().__init__(env)
        if not isinstance(decisions, int) or decisions <= 0:
            raise ValueError("decisions must be a positive integer")
        self.decisions = decisions
        self.decisions_left = decisions

    def step(self, action, decision=True):
        obs, reward, done, info = self.env.step(action)
        self.decisions_left -= decision
        if self.decisions_left == 0:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.decisions_left = self.decisions
        return self.env.reset()


# Make environment using its name
def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def create_folders():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./logs"):
        os.makedirs("./logs")


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