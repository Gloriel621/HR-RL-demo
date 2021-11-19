import copy

import structlog
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from model import PPO
from environment import Environment
from data.data import employees, branches

PRINT_INTERVAL = 10
logger = structlog.get_logger(__name__)


class Trainer:
    def __init__(self):
        self._init_hyperparameters()
        self.employees= employees
        self.branches = branches
        self.env = Environment(employees, branches)
        self.num_employees = self.env.num_employees
        self.num_branches = self.env.num_branches
        self.model = PPO(self.num_employees, self.num_branches)

    def train(self):
        rewards = []
        episode = 0
        while episode <= self.max_episodes:
            self.env.reset()
            state = copy.deepcopy(self.env.state)
            done = self.env.done

            #try:
            while not done:
                for t in range(self.mini_batch_size):

                    prob1 = self.model.pi(torch.from_numpy(self.env.state).float())
                    prob1 = prob1 * torch.from_numpy(np.any(1-self.env.infeasible, axis=1)).float()
                    prob1 = F.normalize(prob1, dim=1, p=1.0)
                    categorical_distribution = Categorical(prob1)
                    employee = categorical_distribution.sample().item()

                    prob2 = self.model.pi2(torch.from_numpy(self.env.state).float())
                    prob2 =  F.normalize(prob2, dim=1, p=1.0)
                    categorical_distribution = Categorical(prob2)
                    branch = categorical_distribution.sample().item()

                    action = (employee, branch)

                    new_state, reward, done = self.env.step(action)
                    self.model.put_data(
                        (
                            copy.deepcopy(state),
                            employee,
                            float(reward/128),
                            copy.deepcopy(new_state),
                            prob1[0][employee].item(),
                            prob2[0][branch].item(),
                            done,
                        )
                    )
                    state = copy.deepcopy(new_state)
                    if done:
                        rewards.append(reward)
                        break
                self.model.train_net()
            # except Exception as e:
            #     logger.info(f"episode : {episode}, reward : {reward}")
            #     logger.error(e)
            #     #torch.save(employee_model.state_dict(), f"hr_ppo_demo_{episode}.pt")
            #     employee_model = PPO(self.num_employees * self.num_branches)
            #     #branch_model = PPO(self.num_branches)
            #     episode = 0
            #else:
            episode += 1

            if episode % PRINT_INTERVAL == 0 and episode != 0:
                print("Episode :{}, avg reward : {:.2f}".format(episode, np.mean(rewards)))
                rewards = []
        torch.save(self.model.state_dict(), f"hr_ppo_demo_{episode}.pt")


    def _init_hyperparameters(self):
        self.max_episodes = 5000
        self.mini_batch_size = 50
