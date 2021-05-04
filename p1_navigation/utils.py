from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def watch(agent, env, brain_name):
    """ Watch the agent play one episode."""
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state, eps=-1)  # eps=-1 to ensure always using the max, greedy action
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print("Score: {}".format(score))


def interact(agent, env, brain_name, n_episodes=2000, min_score=13.0, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Double Deep Q-Learning with experience replay.
       Agent-environment interaction loop and scores.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        min_score (float): Average score at which to stop training
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    best_avg_reward = np.NINF
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        while True:
            action = agent.act(state, eps)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score > best_avg_reward:
            print(f"Score improved from {best_avg_reward} to {mean_score}. Saving the model.")
            best_avg_reward = mean_score
            torch.save(agent.qnetwork_local.state_dict(), 'best_model.pth')
            if mean_score >= min_score:
                print(
                    '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, mean_score))
                torch.save(agent.qnetwork_local.state_dict(), 'top_model.pth')
                break
    return scores, best_avg_reward
