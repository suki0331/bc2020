import gym
env = gym.make("Breakout-v0")
# env = gym.wrappers.Monitor(env, 'recording', force=True)
observation = env.reset()

for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close