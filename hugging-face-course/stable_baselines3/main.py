from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def env_peak(env_name):
    env = gym.make(env_name)
    env.reset()

    print("Observation Space")
    print(f"There are {env.observation_space.shape} variables in observation space")
    print(f"Example: {env.observation_space.sample()}")

    print("Action Space")
    print(f"There are {env.action_space.shape} variables in action space")
    print(f"Example: {env.action_space.sample()}")


def train_model(env_name, model_name):
    # create a stack of 16 envs
    env = make_vec_env(env_name, n_envs=16)

    # instantiate the agent with some parameters for faster convergence
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1
    )

    model.learn(total_timesteps=1_000_000)
    model.save(model_name)


def eval_agent(env_name, model):
    eval_env = Monitor(gym.make(env_name))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")


if __name__ == "__main__":
    env_peak("LunarLander-v2")

    # To train model and save trained parameters into zip file
    train_model("LunarLander-v2", "ppo-LunarLander-v2")

    # To load parameters file
    model = PPO.load("ppo-LunarLander-v2", print_system_info=True)

    # To evaluate the loaded model
    eval_agent("LunarLander-v2", model)

