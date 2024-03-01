import torch.nn as nn

class DQN(nn.Module):
    # estimates the Q values from state and action
    def __init__(self, state_dim, hidden_size, nb_actions):
        super().__init__()
        self.state_dim = state_dim
        self.nb_neurons = hidden_size
        self.nb_actions = nb_actions

        self.main = nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                          nn.ReLU(),
                          nn.Linear(self.nb_neurons, self.nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(self.nb_neurons, self.nb_actions))

    def forward(self, input):
        return self.main(input)
    