import gymnasium as gym
import time
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import RecordVideo

# Virtual Display Handling for Windows Compatibility
import contextlib


@contextlib.contextmanager
def dummy_display():
    yield


# Use the dummy display context
display = dummy_display()


def main():
    # ------------------------------
    # Initialize Virtual Display (Windows-friendly approach)
    # ------------------------------
    with display:
        # ------------------------------
        # Load the Trained Model
        # ------------------------------
        model = DQN.load("models/dqn_model.zip")

        # ------------------------------
        # Environment Setup for Evaluation with Video Recording
        # ------------------------------
        env_id = "ALE/Boxing-v5"
        # Use render_mode "rgb_array" to capture frames for video recording.
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)

        # Wrap with RecordVideo to record every episode (videos will be saved to the specified folder)
        env = RecordVideo(env, video_folder="videos/",
                          episode_trigger=lambda episode_id: True)

        # Create a separate environment instance for real-time rendering
        render_env = gym.make(env_id, render_mode="human")
        render_env = AtariWrapper(render_env)

        # ------------------------------
        # Playing with the Agent using a Greedy Policy
        # ------------------------------
        episodes = 5  # Number of evaluation episodes

        for ep in range(episodes):
            obs, info = env.reset()
            render_env.reset()  # Reset the render environment as well
            done = False
            episode_reward = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                # Use the separate render environment to display the game
                render_obs, render_reward, render_terminated, render_truncated, render_info = render_env.step(
                    action)
                time.sleep(0.1)  # Slow down for visualization

                if done or truncated:
                    break
            print(f"Episode {ep+1} Reward: {episode_reward}")

        env.close()
        render_env.close()


if __name__ == '__main__':
    main()
