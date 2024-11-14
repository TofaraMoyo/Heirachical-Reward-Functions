import argparse
from neptune_utils import init_neptune_run
from train_utils import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for TLA model.")
    parser.add_argument("--env_name", default="Pendulum-v1", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--eval", )
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args()

    run = init_neptune_run()
    train(run=run,
          seed=args.seed,
          env_name=args.env_name)
