import copy

import structlog
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from model import PPO
from environment import Environment
from data.data import employees, branches

PRINT_INTERVAL = 50
logger = structlog.get_logger(__name__)


class Trainer:
    def __init__(self):
        self._init_hyperparameters()
        self.employees= employees
        self.branches = branches
        self.env = Environment(self.employees, self.branches)
        self.num_employees = self.env.num_employees
        self.num_branches = self.env.num_branches
        self.employee_model = PPO(self.num_employees * self.num_branches)
        #self.branch_model = PPO(self.num_branches)

    def train(self):
        rewards = []
        episode = 0
        while episode <= self.max_episodes:
            self.env.reset()
            state = copy.deepcopy(self.env.state)
            done = self.env.done

            try:
                while not done:
                    for t in range(self.mini_batch_size):
                        prob = self.employee_model.pi(torch.from_numpy(state).float())
                        prob = prob.reshape(self.num_employees, self.num_branches)
                        prob = prob * torch.from_numpy(1 - self.env.infeasible).float()
                        prob = F.normalize(prob, dim=1, p=1.0)
                        prob = prob.reshape(-1, self.num_employees * self.num_branches)

                        categorical_distribution = Categorical(prob)
                        action = categorical_distribution.sample().item()
                        new_state, reward, done = self.env.step(action)
                        self.employee_model.put_data(
                            (
                                copy.deepcopy(state),
                                action,
                                float(reward),
                                copy.deepcopy(new_state),
                                prob[0][action].item(),
                                done,
                            )
                        )
                        state = copy.deepcopy(new_state)
                        if done:
                            rewards.append(reward)
                            break
                    self.employee_model.train_net()
            except Exception as e:
                logger.info(f"episode : {episode}")
                logger.error(e)
                #torch.save(self.employee_model.state_dict(), f"hr_ppo_demo_{episode}.pt")
                self.employee_model = PPO(self.num_employees * self.num_branches)
                #self.branch_model = PPO(self.num_branches)
                episode = 0
            else:
                episode += 1

                if episode % PRINT_INTERVAL == 0 and episode != 0:
                    print("Episode :{}, avg reward : {:.2f}".format(episode, np.mean(rewards)))
                    rewards = []
        torch.save(self.employee_model.state_dict(), f"hr_ppo_demo_{episode}.pt")

    def _init_hyperparameters(self):
        self.max_episodes = 10000
        self.mini_batch_size = 50
