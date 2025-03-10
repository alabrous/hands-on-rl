import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from huggingface_hub import login
from huggingface_sb3 import push_to_hub
import wandb
from wandb.integration.sb3 import WandbCallback
import time

def main(save_and_push=False):

    config = {
        "policy": "MlpPolicy",
        "env_name": "CartPole-v1",
        "total_timesteps": 25000
    }

    experiment_name = f"A2C_{int(time.time())}"

    run = wandb.init(
        name=experiment_name,
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        env = gym.make("CartPole-v1",render_mode="rgb_array")
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])

    #env = VecVideoRecorder(env, "videos",
        #record_video_trigger=lambda x: x % 2000 == 0, video_length=200)  # record videos

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(
        gradient_save_freq=100,
        #model_save_path=f"models/{run.id}",
        verbose=2,
    ))

    if save_and_push:
        model.save("a2c_cartpole")

        login()

        push_to_hub(repo_id="AxelLabrousse/a2c_cartpole", filename="a2c_cartpole.zip", commit_message="Add CartPole model trained with A2C")


if __name__ == "__main__":
    main(save_and_push=False)