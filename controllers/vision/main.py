from gymnasium.envs.registration import register
import gymnasium as gym

register (
    id="WebotEnv-v0",
    entry_point = "vision:WebotEnv"
)

env = gym.make("WebotEnv-v0")
state = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    if done:
        break
    state = next_state
env.close()