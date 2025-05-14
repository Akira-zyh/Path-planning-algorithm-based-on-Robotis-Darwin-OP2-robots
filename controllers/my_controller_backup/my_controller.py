from robot_supervisor import NavigationRobotSupervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from controller import Supervisor
import os

env = NavigationRobotSupervisor(description="", seed=2)

controller_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(controller_dir, "training_logs_deepbots_darwin/")
model_path = os.path.join(LOG_DIR, "final_model.zip")
if os.path.exists(model_path):
    # print(f"Loading model from {model_path}")
    # model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=LOG_DIR)
    print("Creating new model")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
else:
    print("Creating new model")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=LOG_DIR,
    name_prefix='model',
)

print("Starting training")

try:
    model.learn(total_timesteps=5000000, callback=[checkpoint_callback])
except Exception as e:
    print("Error occured")
    env.close()
    raise

final_path = os.path.join(LOG_DIR, "final_model.zip")
model.save(final_path)
print(f"Saved model to {final_path}")
print("Finished training")
while env.step(32) != -1:
    pass

env.close()