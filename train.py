import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback

# ------------------------------
# Custom Callback to Log Training Details and Metrics
# ------------------------------


class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_length += 1
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_info = info["episode"]
                self.episode_rewards.append(episode_info["r"])
                self.episode_lengths.append(episode_info["l"])
                print(
                    f"Episode Reward: {episode_info['r']:.2f} | Length: {episode_info['l']}")
        return True

def main():
    
    # ------------------------------
    # Environment Setup
    # ------------------------------
    # Using the Atari Boxing environment (ALE namespace)
    env_id = "ALE/Boxing-v5"
    env = gym.make(env_id, render_mode=None)
    env = AtariWrapper(env)
    
    # ------------------------------
    # Define Hyperparameters (Hyperparameter Set I)
    # ------------------------------
    learning_rate = 1e-4
    gamma = 0.99
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay = 1000000  # timesteps over which epsilon decays
    
    # Using a CNN-based policy for visual input
    policy = "CnnPolicy"
    
    # Initialize the DQN agent with the above parameters
    model = DQN(
        policy,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        verbose=1,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        exploration_fraction=epsilon_decay / 1_000_000,
    )
    
    # ------------------------------
    # Training the Agent
    # ------------------------------
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=TrainingLogger())
    
    # Save the trained model 
    model.save("models/dqn_model.zip")
    print("Model saved as dqn_model.zip")
    env.close()
    
if __name__ == '__main__':
    main()
