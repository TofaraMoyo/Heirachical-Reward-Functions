hyperparameters = {
    'Default': {
        'timestep': 1,
        'frame_skip': 1,
        'type': 'mujoco',
        'max_timesteps': 1000000,
        'eval_freq': 5000,
        'start_timesteps': 1000,
        'discount': 0.99,
        'tau': 0.005,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'policy_freq': 2,
        'replay_size': 1000000,
        'expl_noise': 0.1,
        'batch_size': 256
    },
    'Pendulum-v1': {
        'timestep': 0.05,
        'frame_skip': 1,
        'type': 'cc',
        'eval_freq': 2500,
        'max_timesteps': 30000
    },

    'MountainCarContinuous-v0': {
        'timestep': 1,
        'frame_skip': 1,
        'type': 'cc',
        'start_timesteps': 1000,
        'eval_freq': 5000,
        'max_timesteps': 500000,
        'expl_noise': 0.3,
    },
    'LunarLanderContinuous-v2': {
        'timestep': 1,
        'frame_skip': 1,
        'type': 'cc',
        'start_timesteps': 10000,
        'eval_freq': 2500,
        'max_timesteps': 500000
    },
    'InvertedPendulum-v2': {
        'timestep': 0.02,
        'frame_skip': 2,
        'max_timesteps': 1000000,
    },
    'InvertedDoublePendulum-v2': {
            'timestep': 0.01,
            'frame_skip': 5,
            'max_timesteps': 1000000
        },

    'Walker2d-v2': {
                'timestep': 0.002,
                'frame_skip': 4,
                'max_timesteps': 1000000,
                'start_timesteps': 20000
    },
    'Hopper-v2': {
                'timestep': 0.002,
                'frame_skip': 4,
                'max_timesteps': 1000000,
                'start_timesteps': 20000
    },
    'Ant-v2': {
                    'timestep': 0.01,
                    'frame_skip': 5,
                    'max_timesteps': 1000000,
                    'start_timesteps': 20000
        },
    'HalfCheetah-v2': {
                    'timestep': 0.01,
                    'frame_skip': 5,
                    'max_timesteps': 1000000,
                    'start_timesteps': 20000
        },
    'myoFingerPoseFixed-v0':{
        'type':'myo',
        'timestep': 0.002,
        'frame_skip': 10,
        'max_timesteps': 1000000,
        'start_timesteps': 20000
    },
    'myoChallengeRelocateP1-v0':{
            'type':'myo',
            'timestep': 0.002,
            'frame_skip': 10,
            'max_timesteps': 1000000,
            'start_timesteps': 20000,
    },
    'MsPacmanDeterministic-v4': {
        'epsilon_timesteps': 1_000_000,
        'final_epsilon': 0.01,
        "initial_epsilon": 1.0,
        "episodes": 20000,
        "training_steps":50_000_000,
        "max_env_time_steps": 10000,
        "eval_every_n_steps": 10000,
        "learning_starts": 50000,
        "train_freq":4
    }

}

hyperparametersSAC = {
    'Default': {
        'timestep': 1,
        'frame_skip': 1,
        'alpha':0.2,
        'hidden_size':256,
        'type': 'mujoco',
        'max_timesteps': 1000000,
        'eval_freq': 5000,
        'start_timesteps': 10000,
        'updates_per_step':1,
        'target_update_interval':1,
        'discount': 0.99,
        'tau': 0.005,
        'lr':0.0003,
        'noise_clip': 0.5,
        'policy_freq': 2,
        'replay_size': 1000000,
        'expl_noise': 0.1,
        'batch_size': 256,
        'max_episode_steps': 1000
    },
    'Hopper-v2': {
        'timestep': 0.002,
        'frame_skip': 4,
    },
    'Walker2d-v2': {
            'timestep': 0.002,
            'frame_skip': 4,
    },
    'Ant-v2': {
        'timestep': 0.01,
        'frame_skip': 5,
        'max_timesteps': 5000000,
    },
    'HalfCheetah-v2': {
            'timestep': 0.01,
            'frame_skip': 5,
            'max_timesteps': 5000000,
    },
    'Ant-v3': {
        'timestep': 0.01,
        'frame_skip': 5,
    },
    'Humanoid-v2':{
        'timestep': 0.003,
        'frame_skip': 5,
        'max_timesteps': 10000000
    },
    'Pendulum-v1': {
        'timestep': 0.05,
        'frame_skip': 1,
        'type': 'cc',
        'eval_freq': 2500,
        'max_timesteps': 30000,
        'max_episode_steps':250,
        'start_timesteps': 1000,
    },
    'LunarLanderContinuous-v2': {
        'timestep': 1,
        'frame_skip': 1,
        'type': 'cc',
        'start_timesteps': 10000,
        'eval_freq': 2500,
        'max_timesteps': 500000,
    },
    'InvertedPendulum-v2': {
        'timestep': 0.02,
        'frame_skip': 2,
        'max_timesteps': 1000000,
    },
    'InvertedDoublePendulum-v2': {
        'timestep': 0.01,
        'frame_skip': 5,
        'max_timesteps': 1000000
    },

}


def get_hyperparameters(env_name, type="TD3"):

    if type == "TD3":

        obj = hyperparameters['Default']
        for key in hyperparameters[env_name].keys():
            obj[key] = hyperparameters[env_name][key]
        return obj
    elif type == "SAC":
        obj = hyperparametersSAC['Default']
        for key in hyperparametersSAC[env_name].keys():
            obj[key] = hyperparametersSAC[env_name][key]
        return obj