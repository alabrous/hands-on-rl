import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from huggingface_hub import login
from huggingface_sb3 import push_to_hub
import wandb
from wandb.integration.sb3 import WandbCallback
import time

import panda_gym

def main(save_and_push=False):

    config = {
        "policy": "MultiInputPolicy",
        "env_name": "PandaReachJointsDense-v3",
        "total_timesteps": 5000
    }

    experiment_name = f"DDPG_{int(time.time())}"

    run = wandb.init(
        name=experiment_name,
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        env = gym.make(config["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])

    model = DDPG(config["policy"], env, replay_buffer_class = HerReplayBuffer, replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy= "future",
    ), verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(
        gradient_save_freq=100,
        #model_save_path=f"models/{run.id}",
        verbose=2,
    ))

    if save_and_push:
        model.save("ddpg_reachjointsdense2")

        login()

        push_to_hub(repo_id="AxelLabrousse/ddpg_reachjointsdense", filename="ddpg_reachjointsdense2.zip", commit_message="Add Reach Model with joints trained with DDPG")


if __name__ == "__main__":
    main(save_and_push=True)