import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(10000000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()  # https://github.com/openai/gym/issues/893