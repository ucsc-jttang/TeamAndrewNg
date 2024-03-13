# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.MainAgent import MainAgent
from Agents.RedAgent import RedAgent
from Agents.RedPPOAgent import RedPPOAgent
import random

MAX_EPS = 1
agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    

    
    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    

    # Load blue agent
    blue_agent = WrappedBlueAgent
    #this must be loaded from a specified location
    # red_agent = RedPPOAgent(input_dims=52, action_space=[i for i in range(888)], lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=True, ckpt='../Models/checkpoints/200.pth',
    #              deterministic=False, training=True, start_actions=[])
    red_agent = RedAgent(restore=True, ckpt='/home/ubuntu/233/Models/checkpoints/62800.pth')
    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")
    

    for num_steps in [30]:
        observation = env.reset()

        # action_space = wrapped_cyborg.get_action_space(agent_name)
        action_space = env.get_action_space(agent_name)
        total_reward = []
        actions = []
        for i in range(MAX_EPS):
            r = []
            a = []
            for j in range(num_steps):
                action = red_agent.get_action(observation, action_space) 
                observation, rew, done, info = env.step(action)
                r.append(rew)
                a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
            total_reward.append(sum(r))
            actions.append(a)
            observation = env.reset()

        print(total_reward)
        print(actions)

            
