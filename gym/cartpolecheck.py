import gym
env = gym.make('Pendulum-v0')


# for i_episode in range(20):
observation = env.reset()
# for t in range(100):
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(done,reward)
    # if done:
    #     print("Episode finished after {} timesteps".format(t+1))
    #     break
env.close()