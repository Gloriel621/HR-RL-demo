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
        self.env = Environment(self.employees, self.branches)
        self.num_employees = self.env.num_employees
        self.num_branches = self.env.num_branches
        self.employee_model = PPO(self.num_employees * self.num_branches, self.num_employees)
        self.branch_model = PPO(self.num_employees * self.num_branches, self.num_branches)

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

                    employee = self._sample_from_network(self.employee_model)
                    while np.sum(1 - self.env.infeasible[employee]) == 0:
                        employee = self._sample_from_network(self.employee_model)
                    
                    branch = self._sample_from_network(self.branch_model)



                    prob2 = self.branch_model.pi(torch.from_numpy(state).float())
                    prob2 = prob2 * torch.from_numpy(1 - self.env.infeasible[employee]).float()
                    prob2 = F.normalize(prob2, dim=1, p=1.0)
                    distribution2 = Categorical(prob2)
                    branch = distribution2.sample().item()

                    # prob = prob.reshape(self.num_employees, self.num_branches)
                    # prob = prob * torch.from_numpy(1 - self.env.infeasible).float()
                    # prob = F.normalize(prob, dim=1, p=1.0)
                    # prob = prob.reshape(-1, self.num_employees * self.num_branches)
                    # categorical_distribution = Categorical(prob)
                    
                    # action = categorical_distribution.sample().item()
                    action = (employee, branch)

                    new_state, reward, done = self.env.step(action)
                    self.employee_model.put_data(
                        (
                            copy.deepcopy(state),
                            employee,
                            float(reward/128),
                            copy.deepcopy(new_state),
                            prob1[0][employee].item(),
                            done,
                        )
                    )
                    self.branch_model.put_data(
                        (
                            copy.deepcopy(state),
                            branch,
                            float(reward/128),
                            copy.deepcopy(new_state),
                            prob2[0][branch].item(),
                            done,
                        )
                    )
                    state = copy.deepcopy(new_state)
                    if done:
                        rewards.append(reward)
                        break
                self.employee_model.train_net()
                self.branch_model.train_net()
            # except Exception as e:
            #     logger.info(f"episode : {episode}, reward : {reward}")
            #     logger.error(e)
            #     #torch.save(self.employee_model.state_dict(), f"hr_ppo_demo_{episode}.pt")
            #     self.employee_model = PPO(self.num_employees * self.num_branches)
            #     #self.branch_model = PPO(self.num_branches)
            #     episode = 0
            #else:
            episode += 1

            if episode % PRINT_INTERVAL == 0 and episode != 0:
                print("Episode :{}, avg reward : {:.2f}".format(episode, np.mean(rewards)))
                rewards = []
        torch.save(self.employee_model.state_dict(), f"hr_ppo_demo_{episode}.pt")
    
    def _sample_from_network(self, network:PPO, mode = "employee"):

        prob = network.pi(torch.from_numpy(self.state).float())
        if mode == "branch":
            prob = prob * torch.from_numpy(1 - self.env.infeasible[employee]).float()
        prob = F.normalize(prob, dim=1, p=1.0)
        distribution = Categorical(prob)
        
        sample = distribution.sample().item()

        return sample


    def _init_hyperparameters(self):
        self.max_episodes = 5000
        self.mini_batch_size = 50
