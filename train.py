from collections import deque
import gym
from numpy import double
import torch
from agent import PPO_Agent, REINFORCE_Agent
import argparse

def train(agent,env,num_episodes,num_steps,log_every=20,render_from=500):
    running_avg_reward = deque(maxlen=100)
    for episode in range(1,num_episodes+1):
        obs = env.reset()
        reward_total = 0
        for step in range(num_steps):
            if episode>render_from:
                env.render()
            action_dist = agent.act(torch.tensor(obs,dtype=torch.float32))
            action = action_dist.sample()
            obs , reward, done, info = env.step(action.numpy())
            reward_total+=reward
            agent.cache(reward,action_dist.log_prob(action))
            if done:
                if episode % log_every ==0:
                    print("Episode {} finished after {} timesteps with a total reward of {} | Running Average: {}".format(episode,step+1,reward_total,sum(running_avg_reward)/len(running_avg_reward)))
                break
        agent.update_model()
        running_avg_reward.append(reward_total)


    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a agent on gym environments')

    parser.add_argument('--gym_env', "-g", default="CartPole-v1", type=str,
                        help='The name of the gym environment')

    parser.add_argument('--hidden_size', "-hs", default=32, type=int,
                        help='size of the hidden layer')

    parser.add_argument('--gamma', "-gm", default=0.999, type=float,
                        help='The discount factor used by the agent')

    parser.add_argument('--learning_rate', "-lr", default=0.001, type=float,
                        help='The learning rate used by the optimizer')
    
    parser.add_argument('--use_PPO', "-up",
                        action='store_true', help='Use PPO algorithm instead of REINFORCE')

    args = parser.parse_args()

    env = gym.make(args.gym_env)

    if not args.use_PPO:
        agent = REINFORCE_Agent(env.observation_space.shape[0],args.hidden_size,env.action_space.n,True,args.gamma,args.learning_rate)

    train(agent,env,3000,201)