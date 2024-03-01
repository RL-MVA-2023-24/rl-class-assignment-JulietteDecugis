from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import joblib
import torch
import torch.nn as nn
from DQN import DQN
from ReplayBuffer import ReplayBuffer
from tqdm import tqdm

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProjectAgent:
    def __init__(self) -> None:
        self.state_dim = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n
        self.gamma = 0.9

        self.buffer_size = 1000000
        self.epsilon_min = 0.05
        self.epsilon_max = 1
        self.epsilon_delay = 20 # step in horizon where eps starts to decay
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/1000
        self.gamma = 0.95
        self.tau = 0.005
        self.path = "policy_network.pth"
        
        self.criterion = torch.nn.MSELoss()
        self.batch_size = 32
        self.nb_neurons = 24
        self.losses = []

        self.memory = ReplayBuffer(self.buffer_size, device)
        self.policy_network = DQN(self.state_dim, self.nb_neurons, self.nb_actions).to(device)
        self.target_network = DQN(self.state_dim, self.nb_neurons, self.nb_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = 0.01)
    
    def greedy_action(self, state):
        device = next(self.policy_network.parameters()).device
        with torch.no_grad():
            Q = self.policy_network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(range(self.nb_actions), 1)[0]
        else:
            # gredy action based on current Q estimate
            return self.greedy_action(observation)

    def save(self, path):
        # save both networks or just main? CHECK
        torch.save(self.policy_network.state_dict(), path)
        self.path = path

    def load(self):
        if self.path is not None:
            self.policy_network = DQN(self.state_dim, self.nb_neurons, self.nb_actions).to(device)
            self.policy_network.load_state_dict(torch.load(self.path)).to(device)            
        # self.train(epochs=10, max_episode=1000)
        # self.save('policy_network.pth')
    
    def optimize_model(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.policy_network(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss)

    def train(self, epochs, max_episode):
        max_episode_length = 200
        episode_return = []
        episode_cum_reward = 0
        state, _ = env.reset()

        for episode in tqdm(range(max_episode)):
            epsilon = self.epsilon_max
            for t in range(max_episode_length):
                if t > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
                
                if np.random.rand() < epsilon:
                    action = self.act(state, use_random=True)
                else:
                    action = self.act(state, use_random=False)
                
                # step
                next_state, reward, done, _, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward

                for _ in range(epochs):
                    self.optimize_model()
                
                # update target network with EMA
                target_state_dict = self.target_network.state_dict()
                model_state_dict = self.policy_network.state_dict()
                for key in model_state_dict:
                    target_state_dict[key] = self.tau*model_state_dict[key] + (1-self.tau)*target_state_dict[key]
                self.target_network.load_state_dict(target_state_dict)

                # next transition
                if done:
                    state, _ = env.reset()
                    episode_return.append(episode_cum_reward)
                    episode_cum_reward = 0
                    if episode % 10 == 0:
                        print("Episode ", '{:3d}'.format(episode), 
                              ", epsilon ", '{:6.2f}'.format(epsilon), 
                              ", batch size ", '{:5d}'.format(len(self.memory)), 
                              ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                              sep='')
                        print('Losses: ', np.mean(self.losses))
                    break
                else:
                    state = next_state
        return episode_return