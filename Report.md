[//]: # (Image References)

[plot]: rewardsplot.png "rewardsplot"
[replay]: bananathief.gif "bananathief"


# Deep Q Learning Navigation: Banana Picker

In this project, an agent was trained to navigate an environment to pick yellow bananas while avoiding blue ones. The 
reward system is such that the agent receives a reward of +1 for picking a yellow banana and -1 for picking a blue one.
The agent is considered to have been trained successfully if it gets an average cumulative score of +13.

The state space contains the agent's velocity and ray based perception of objects along its forward direction. The state
space has 37 dimensions. The action space on the other hand has length of four, with the following possible actions: 
move forward, move backward, turn left and turn right.

### Neural Network

A neural network was used to form a model that learns over time, the best action to take when in a particular state. The 
neural network used here has three fully connected layers and has the following details/parameters:
- **`state size`** - the dimension of the state space, used to indicate the number of input features in the first layer
- **`arbitrary numbers`** - these are used to provide the number of output and input features into the layers  
- **`action size`** - the dimension of the action space, used to indicate the number of output features of the last layer
- **`dropout probability`** - the probability with which neurons should be zeroed in respective layers 

The network is a feed forward network and applies drop out and ReLU function activations to predict the best actions. This
is used in training the agent, where, instead of choosing random actions and getting rewards, it uses the model to select
an action using an epsilon greedy policy. The model uses the *Adam* optimizer, with a learning rate of **0.0005**, and uses
the **mean square error** algorithm to calculate the loss between the expected and the computed values.

### Agent Training

The agent trains by interacting with the environment. Implementing the Markov decision process, at every time step, the
agent is presented with a state of the environment, takes an action, gets rewarded and then ends up in the next state.
The agent employs a strategy of learning from experiences, this will be clarified later.

The agent is trained over a number of episodes and performs the following actions per episode:
1. With the environment state reset at the beginning of an episode, the agent selects an action to take at that state. 
   The agent selects an epsilon greedy action to take from a set of actions returned by the neural network. The epsilon greedy parameters that worked
   best here were to start with the value of **1.0** (where the agent acts randomly), decay at a rate of **0.99525** per epsiode
   until the epsilon reaches a minimum **0.01** (where the agent acts greedily towards taken the action that yields the best
   cumulative reward).
   
2. The agent then takes the selected epsilon greedy action, following which the environment presents it with a reward, 
   the next state, and an indicator of whether the episode is done or not. (An episode is done when the agent gets a negative
   cumulative reward.)
   
3. Following this, the agent adds its recent experience to a memory buffer of experiences. The memory of experiences is
   added to until it reaches a certain batch size (**64** in our case). When the memory size is at least equal to the 
   batch size, random experiences are sampled from the memory buffer to be used to train the neural network further 
   (calculating the loss and applying backward propagation) and to update the expected Q values. The discount factor 
   that worked for updating the expected Q values was **0.99**. 


### Results
Targeting an average score of +13.0 over 100 consecutive episodes, the agent was able to be trained to achieve this after
**490** episodes of training. As the number of episodes increased, so did the average score, as can be seen below:

![rewardsplot][plot]

The model weights of the agent were stored. Testing the agent with the saved model yielded the following performance:
![bananathief][replay]


### Potential Areas of Improvement
Although the agent performs well, it can still do better. For example, as can be seen in the video, the agent becomes
pointless about what to do when there are no yellow bananas nearby, therefore turning left and right to try locate nearby
bananas instead of taking a step forward.

Potential improvements to fix such include, but are not limited to:
- Adding more layers to the neural network architecture
- Experimenting with different model optimizers and learning rates
- Experimenting with double Q learning algorithm to minimize the overestimation of action values
- Implementing prioritized experience replay algorithm, which will replay experiences which are of more significance