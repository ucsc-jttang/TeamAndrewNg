# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import time
import torch
import numpy as np
import os


from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.RedAgent import RedAgent
from Agents.RedPPOAgent import RedPPOAgent
import random

MAX_EPS = 100
agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):



    agent = RedPPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions)

    running_reward, time_step = 0, 0

    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state)
            try:
                state, reward, done, _ = env.step(action)
                agent.store(reward, done)
            except:

                agent.store(-100, done)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            running_reward += reward

        agent.end_episode()

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    

    
    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'


    # Load blue agent
    blue_agent = WrappedBlueAgent
    # red_agent = RedPPOAgent()
    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")
    action_space_size = env.get_action_space('Red')
    action_space = [i for i in range(action_space_size)]


    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # change checkpoint directory
    folder = 'checkpoints'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # CYBORG = CybORG(PATH, 'sim', agents={
    #     'Red': B_lineAgent
    # })
    # env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]

    print_interval = 50
    save_interval = 200
    max_episodes = 100000
    max_timesteps = 100
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002

    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval, start_actions=[])

            #training function must save model to a location

            # red_agent.train(...) # CSE233 Project: uncoment when you implement red agent training
            

