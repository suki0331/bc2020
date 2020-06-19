import gym

env_name = 'Breakout-v0'
env = gym.make(env_name)

frames = []
env.reset()
done= False
for _ in range(300):
    #print(done)
    frames.append(env.render(mode='rgb_array'))
    obs, reward, done, _ = env.step(env.action_space.sample())
    if done:
        break
