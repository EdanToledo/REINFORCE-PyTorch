from torch import nn
import torch.nn as nn
import torch.nn.functional as F


class Discrete_Policy(nn.Module):

    def __init__(self, input_size, hidden_size, nb_actions) -> None:
        super(Discrete_Policy, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, nb_actions),
                                        nn.Softmax(dim=-1))
        
    def forward(self,obs):
        return self.policy_net(obs)

        
class Continuous_Policy(nn.Module):

    def __init__(self, input_size, hidden_size, nb_actions) -> None:
        super(Continuous_Policy, self).__init__()
        self.mean_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, nb_actions))
        self.standard_deviation_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_size, nb_actions))
        
    def forward(self,obs):
        return self.mean_net(obs),self.standard_deviation_net(obs)