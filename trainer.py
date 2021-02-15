from dqn_agent import YellowBananaThief
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import torch

file_name = "Banana.app"
env = UnityEnvironment(file_name=file_name)


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


def train(env, agent, target_score=13.0, episodes=2000, eps_start=1.0, eps_min=.01, eps_decay=.995,
          filename="checkpoint.pth"):
    """ This function performs the training of the agent

        :param eps_decay:
        :param eps_min:
        :param eps_start:
        :param env:
        :param filename:
        :param episodes: number of episodes to train the agent
    """
    print("Training has begun ... The target score to reach is {0} in {1} episodes".format(target_score, episodes))
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = eps_start
    target_was_reached = False
    for episode in range(1, episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state (as a numpy array)
        score = 0
        while True:
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        epsilon = max(eps_min, eps_decay * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

        if score >= target_score:
            target_was_reached = True
        if target_was_reached:
            if np.mean(scores_window) >= target_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                             np.mean(scores_window)))
                print("Target score of {0} has been reached. Saving model to file '{1}'".format(target_score, filename))
                torch.save(agent.local_qnet().state_dict(), filename)
                break
    print("Finished training " + "successfully. Max score: {}".format(max(scores)) if target_was_reached else "unsuccessfully")
    return scores


def run(agent_, filename):
    try:
        train(env, agent_, target_score=13.0, eps_decay=.9, filename=filename)
    finally:
        # make sure the environment gets closed regardless of what happens
        env.close()


def test(agent_, filename):
    agent_.qnetwork_local.load_state_dict(torch.load(filename))

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent_.act(state, .0)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()


if __name__ == '__main__':
    agent = YellowBananaThief(state_size, action_size, update_frequency=2)
    filename = "checkpoint.pth"
    train_mode = True
    if train_mode:
        run(agent, filename)
    else:
        test(agent, filename)

