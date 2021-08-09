from model import Discrete_Policy,Continuous_Policy
from torch.distributions import Categorical, Normal
from torch.optim import AdamW
import torch

class PPO_Agent:

    def __init__(self,input_size,hidden_size,nb_actions,discrete=True,discount_factor=0.9,learning_rate=0.0001):
        pass

    def act(self,obs):
        pass

class REINFORCE_Agent:

    def __init__(self,input_size,hidden_size,nb_actions,discrete=True,discount_factor=0.9,learning_rate=0.0001):
        self.discrete = discrete
        if discrete:
            self.policy = Discrete_Policy(input_size,hidden_size,nb_actions)
        else:
            self.policy = Continuous_Policy(input_size,hidden_size,nb_actions)

        self.memory = []
        self.gamma = discount_factor
        
        self.optimizer = AdamW(self.policy.parameters(),learning_rate)

    def act(self,obs):
        if self.discrete:
            action_dist = Categorical(self.policy(obs))

        else:
            mu,dev = self.policy(obs)
            action_dist = Normal(mu,dev)
        
        return action_dist

    def cache(self,reward,log_action_prob):
        self.memory.append((reward,log_action_prob))

    def clear_memory(self):
        self.memory.clear()

    def update_model(self):
        
        returns_to_go = []
        Gt=0
        for (reward,_) in self.memory[::-1]:
            Gt = reward+self.gamma*Gt
            returns_to_go.append(Gt)

        returns_to_go = returns_to_go[::-1]

        losses = []
        for Gt,(reward,log_action_prob) in zip(returns_to_go,self.memory):
            losses.append(-log_action_prob*Gt)
        
        self.optimizer.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        self.optimizer.step()
        self.clear_memory()




