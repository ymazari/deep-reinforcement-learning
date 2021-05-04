from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v2')
num_episodes=20000
agent = Agent(env, num_episodes)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes)
