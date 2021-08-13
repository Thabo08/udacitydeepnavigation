[//]: # (Image References)

[image1]: bananathief.gif "bananathief"

# Deep Q Learning Navigation: Banana Picker

This project implements Deep Q Learning tactics to navigate through an environment that has blue and yellow bananas, 
with the objective of only picking the yellow bananas. The environment simulation is done with Unity. 

The agent that interacts with the environment gets a reward of +1 for picking a yellow banana and a reward of -1 for 
picking a blue one. The agent is considered to have successfully navigated through the environment if it has gained a 
cumulative score of 13 in a single episode.

The agent that navigates the environment employs deep learning methods in learning how to achieve its goal. Written in
Python, the problem was solved in the following ways:


### 1. The Model
For every interaction, the agent observes the current state of the environment, chooses an action and receives a reward
based on the action it took. In this case, the action can be either one of the following: move forward, move backward,
turn left or turn right.

A neural network was used to build a model to choose the best action for a given state. The code implementation can be 
found in the _model.py_ file. The network has 3 fully connected layers, with the ability to choose the preferred number
of units (neurons). The default used here is 64.

As the model gets trained, dropout gets applied to the layers. This is to avoid having neurons with high weights having 
more influence on the output. The default probability of zeroing neurons here is 0.15, but it is configurable.


### 2. The Agent
The code that encompasses the behaviour of the agent as it interacts with the environment is in the _dqn_agent.py_ file. This
is where we define what should happen when the agent interacts with the environment and how it should learn what the best
actions to take are overtime. 

Contained in the _dqn_agent.py_ file are two classes, YellowBananaThief, which encompasses the agent's behavior and learning
algorithm, and ReplayBuffer, which is a dependency of the YellowBananaThief class and is used to enable the agent to replay
previous experiences while learning.


### 3. The Trainer
The code that performs the training is in the utility file _trainer.py_. In this file all required imports are done and,
the agent is trained and tested. The required dependencies can be installed by running:
```shell
   pip install -r requirements.txt
```

The trainer can be used in the following manner when running it straight as python code in a local environment:
```python
   if __name__ == '__main__':
       agent = YellowBananaThief(state_size, action_size, update_frequency=2)
       filename = "checkpoint.pth"
       train_mode = True
       if train_mode:
           run(agent, filename)
       else:
           test(agent, filename)
```

Following training, running the application in test mode should give an output similar to the one below, indicating the
performance of the agent

![bananathief][image1]