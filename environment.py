import os
import copy
from typing import Tuple

import pandas as pd
import numpy as np

from data import Employees, Branches


class Environment:
    def __init__(
        self,
        employees: Employees,
        branches: Branches,
    ):
        self.init_employees = employees
        self.employees = copy.deepcopy(self.init_employees)
        self.init_branches = branches
        self.branches = copy.deepcopy(self.init_branches)

        self.num_employees = len(employees.id)
        self.num_branches = len(branches.id)

    def reset(self):
        self.employees = copy.deepcopy(self.init_employees)
        self.branches = copy.deepcopy(self.init_branches)

        self.done = False
        self.num_steps = 0
        self.reward = 0

        self.state = np.zeros((self.num_employees, self.num_branches))
        self.infeasible = np.zeros((self.num_employees, self.num_branches))

    def step(self, action: int):
        employee_idx = int(action / self.num_branches)
        branch_idx = int(action % self.num_branches)

        is_feasible = self.is_feasible_step(employee_idx, branch_idx)
        if is_feasible:
            self.num_steps += 1
            self.update_current_state_infos(employee_idx, branch_idx)
            if self.num_steps == self.num_required_employees:
                self.done = True
                is_invalid = self.is_invalid_placement()
                if is_invalid:
                    self.reward = -10
                else:
                    self.reward, _ = self.calculate_reward()
        else:
            self.infeasible[employee_idx][branch_idx] = 1

        return self.state, self.reward, self.done

    def render(self):
        state = pd.DataFrame(self.state, index=self.employees.id, columns=self.branches.id)

        return state

    def is_feasible_step(self, employee_idx: int, branch_idx: int):
        """
        # constraints. 매 step마다 조건 확인
        """

        return True

    def update_current_state_infos(self, employee_idx: int, branch_idx: int):
        """
        Update state, infeasible, branches information at every step
        """
        self.state[employee_idx][branch_idx] = 1
        self.infeasible[employee_idx, :] = 1

        employee_rank = self.employees.rank[employee_idx]
        self.branches.num_current_rank[branch_idx][employee_rank] += 1

    def is_invalid_placement(self) -> bool:
        """
        Check whether placement is invalid when all steps are over
        """
        # 지점 정원과 배치된 인원이 다를 시 해당 배치 불가능
        num_required_rank = self.branches.num_required_rank
        num_current_rank = self.branches.num_current_rank
        if not np.array_equal(num_required_rank, num_current_rank):
            return True

        return False

    def calculate_reward(self):
        """
        Calculate reward when all steps are over
        """
        working_months_reward = self.calculate_working_months_reward()
        popular_branch_rotation_reward = self.calculate_popular_branch_rotation_reward()
        preferring_branch_rotation_reward = self.calculate_preferring_branch_rotation_reward()  # noqa
        accessible_branch_reward = self.calculate_accessible_branch_reward()
        remote_placement_priority_reward = self.calculate_remote_placement_priority_reward()  # noqa
        remote_placement_count_reward = self.calculate_remote_placement_count_reward()

        employee_reward = (
            working_months_reward
            + popular_branch_rotation_reward
            + preferring_branch_rotation_reward
            + accessible_branch_reward
            + remote_placement_priority_reward
            + remote_placement_count_reward
        )

        avg_employee_reward_by_branch = np.dot(employee_reward, self.state) / np.sum(self.state, axis=0)

        career_score_reward = self.calculate_career_score_reward()

        branch_reward = (
            avg_employee_reward_by_branch
            + career_score_reward
        )
        reward = np.mean(branch_reward)

        return reward

    def calculate_working_months_reward(self):

        employee_working_months_reward = None
        return employee_working_months_reward

    def calculate_preferring_branch_rotation_reward(self):

        preferring_branch_rotation_reward = None
        return preferring_branch_rotation_reward

    def calculate_popular_branch_rotation_reward(self):

        employee_popular_branch_rotation_reward = None
        return employee_popular_branch_rotation_reward

    def calculate_accessible_branch_reward(self):

        employee_accessible_branch_reward = None
        return employee_accessible_branch_reward

    def calculate_remote_placement_priority_reward(self):

        employee_remote_placement_priority_reward = None
        return employee_remote_placement_priority_reward

    def calculate_remote_placement_count_reward(self):
        
        remote_placement_counts_reward = None
        return remote_placement_counts_reward

    def calculate_career_score_reward(self):
        
        career_score_reward = None
        return career_score_reward
