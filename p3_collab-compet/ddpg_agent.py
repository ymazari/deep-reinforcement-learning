import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import (Actor, Critic)

LR_ACTOR = 1e-3  # learning rate of Actor
LR_CRITIC = 1e-3  # learning rate of Critic
WEIGHT_DECAY = 0.  # L2 weight decay
GAMMA = 0.99  # discount factor
TAU = 1e-3  # soft update parameter
BATCH_SIZE = 512  # batch size to sample from replay buffer
BUFFER_SIZE = int(1e5)  # max size (capacity) of the replay buffer
LEARN_INTERVAL = 20
LEARN_TIMES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent(object):
    def __init__(self, state_size, action_size, env, seed):
        super(DDPGAgent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.env = env
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)

        # initialise local and target Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # initialise local and target Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # initialise the Ornstein-Uhlenbeck noise process
        self.noise = OUNoise((self.num_agents, action_size), seed)

        # Shared Replay Buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0

    def train(self, n_episodes=10000, max_t=1000, print_every=100, min_score=0.5):
        scores_window = deque(maxlen=100)  # mean scores of the last 100 episodes
        moving_avgs = []  # moving averages
        episode_scores = []  # mean scores
        max_score = np.NINF
        for i_episode in tqdm(range(1, n_episodes + 1)):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            scores_list = np.zeros(self.num_agents)
            self.reset()

            for t_step in range(max_t):
                actions = self.act(states, add_noise=True)
                env_info = self.env.step(actions)[self.brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.step(states, actions, rewards, next_states, dones)

                scores_list += rewards
                states = next_states

                if np.any(dones):
                    break

            current_episode_score = np.max(scores_list)
            episode_scores.append(current_episode_score)
            scores_window.append(current_episode_score)
            moving_avgs.append(np.mean(scores_window))

            if i_episode % print_every == 0:
                print(
                    f"\rEpisode {i_episode}\tEpisode Score: {current_episode_score}\tMoving Average Score: {moving_avgs[-1]}")

            if current_episode_score > max_score:
                torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')
                max_score = current_episode_score
                print(f"\rEpisode {i_episode}\tMax Score: {max_score}....checkpoint....")

            if i_episode >= 100 and moving_avgs[-1] >= min_score:
                torch.save(self.actor_local.state_dict(), 'solved_actor.pth')
                torch.save(self.critic_local.state_dict(), 'solved_critic.pth')
                print(
                    f"\n Solved in Episode {i_episode}\tEpisode Score: {current_episode_score}\tMoving Average Score: {moving_avgs[-1]}")
                break

        return episode_scores, moving_avgs

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])

        self.t_step = (self.t_step + 1) % LEARN_INTERVAL
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(LEARN_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))

        self.actor_local.eval()
        with torch.no_grad():
            for i in range(self.num_agents):
                actions[i, :] = self.actor_local(states[i]).cpu().data.numpy()
        self.actor_local.train()

        # Ornstein-Uhlenbeck noise process
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get the actions corresponding to next states and then their Q-values from target critic network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize this loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
            Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params
            ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise(object):
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """ initialise noise parameters """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.seed = torch.manual_seed(seed)
        self.reset()

    def reset(self):
        """reset the internal state to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.array(
            [np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer(object):
    """Replay buffer to store experience tuples"""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
