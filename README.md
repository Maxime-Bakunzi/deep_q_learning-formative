# Deep Q-Learning for Atari Boxing

## Overview
This project implements **Deep Q-Learning (DQN)** to train an agent to play Atari's **Boxing** using **Stable-Baselines3**. The model is trained in Google Colab due to memory constraints and runs in a Jupyter Notebook for training and evaluation. A recorded video of the agent playing 5 episodes is included to showcase its performance.

## Repository Structure
```
deep_q_learning-formative/
│── train.py                # Script to train the DQN model
│── play.py                 # Script to run the trained model and record gameplay
│── requirements.txt        # Required dependencies
│
├── models/                 # Folder containing trained models
│   ├── dqn_model.zip       # Selected trained model
│   ├── dqn_model2.zip      # Alternative model
│   ├── dqn_model3.zip      # Alternative model
│
├── videos/                 # Folder containing gameplay recordings
│   ├── rl-video-episode-0.mp4       # First episode
│   ├── rl-video-episode-1.mp4       # Second episode
│   ├── rl-video-episode-2.mp4       # Third episode
│   ├── rl-video-episode-3.mp4       # Fourth episode
│   ├── rl-video-episode-4.mp4       # Fifth episode
│   ├── merged_video.mp4    # All five episodes combined
│
├── notebook/               # Jupyter notebook used in Colab
│   ├── deep_q_learning.ipynb  # Notebook for training and evaluation
```

## Environment & Setup
### Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Training Script
To train the agent using DQN, run:
```bash
python train.py
```
This will create a trained model inside the `models/` folder.

### Running the Agent (Playing Episodes)
Once trained, you can test the model and record gameplay:
```bash
python play.py
```
The video recordings will be saved in the `videos/` folder.

## Training & Evaluation Process
### **Training (Google Colab)**
- The training process was run in **Google Colab** due to memory limitations.
- We trained different models with different hyperparameters to evaluate performance.
- The final selected model is **dqn_model.zip**, which provided the best results.

### **Playing & Recording Episodes**
- The `play.py` script loads the trained model and plays **5 episodes**.
- The environment was rendered in **Google Colab** using a virtual display.
- The gameplay of all five episodes was recorded and merged into a single video.
- The video file **merged_video.mp4** is included in the repository.

## **Model Comparison & Performance**
The table below compares the different trained models:

| Model           | Hyperparameters & Policy                                                                                                   | Observed Performance                                                                                                                                                                                                                                                                         |
|-----------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **dqn_model.zip**   | **Policy:** CnnPolicy<br>**Learning Rate:** 1e-4<br>**Gamma:** 0.99<br>**Batch Size:** 32<br>**Epsilon:** 1.0 → 0.02 (decay over 1e6 steps)<br>**Timesteps:** 500,000 | The agent steadily improved with average episode rewards rising to around 7–8. It showed stable learning with moderate episode lengths (~441 steps). The CNN-based policy efficiently processed visual input, making it a solid baseline for the Boxing environment.               |
| **dqn_model2.zip**  | **Policy:** MlpPolicy<br>**Learning Rate:** 0.001<br>**Gamma:** 0.99<br>**Batch Size:** 32<br>**Epsilon:** 1.0 → 0.01 (decay over 0.98e6 steps)<br>**Timesteps:** 100,000 | The MLP-based model struggled compared to the CNN version, with average rewards mostly in the negative range (around -10 to -15). Despite faster initial learning due to a higher learning rate, the performance was inconsistent and overall subpar for this visually complex task.          |
| **dqn_model3.zip**  | **Policy:** CnnPolicy<br>**Learning Rate:** 0.002<br>**Gamma:** 0.99<br>**Batch Size:** 64<br>**Epsilon:** 1.0 → 0.05 (decay over 0.99e6 steps)<br>**Timesteps:** 110,000 | The third model, with a higher learning rate and larger batch size, showed mixed results. While some episodes had promising rewards, the overall performance averaged around -3, indicating that the agent was still struggling to consistently score high. Further tuning may help. |

### **Why We Chose dqn_model.zip**
- It had the most **consistent performance** across episodes.
- The CNN-based policy worked well for the **high-dimensional input** of the Atari game.
- The model's **reward improvements** were stable, showing better adaptation to the environment.

## Gameplay Video (Agent Playing 5 Episodes)
We recorded 5 full episodes of the trained agent playing **Atari Boxing**. The merged video file is available in the repository:

📹 **Watch the gameplay video:** [Click to Watch](videos/merged_video.mp4)

## Final Thoughts
This project successfully implemented **Deep Q-Learning** to train an agent for Atari Boxing. The selected model **dqn_model.zip** showed the best performance, and the gameplay videos demonstrate its ability to interact effectively with the environment.

Future improvements can include:
- **Longer training** for better performance.
- **Hyperparameter tuning** for more stability.
- **Exploring other reinforcement learning algorithms** like PPO or A3C.

---


