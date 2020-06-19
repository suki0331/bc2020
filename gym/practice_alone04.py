from gym import spaces
import gym

env = gym.make('CartPole-v0')
env.reset()

space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    print(x)
env.close()

# env = gym.make("CartPole-v0")
# env.render()