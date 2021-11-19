import copy

import pandas as pd
import numpy as np

from data.data import employees, branches


class Environment:
    def __init__(
        self,
        employees: employees,
        branches: branches,
    ):
        self.init_employees = employees
        self.employees = copy.deepcopy(self.init_employees)
        self.init_branches = branches
        self.branches = copy.deepcopy(self.init_branches)

        self.num_required_employees = len(employees)

        self.num_employees = len(employees)
        self.num_branches = len(branches)

    def reset(self):
        self.employees = copy.deepcopy(self.init_employees)
        self.branches = copy.deepcopy(self.init_branches)

        self.done = False
        self.num_steps = 0
        self.reward = 0

        self.state = np.zeros((self.num_employees, self.num_branches))
        self.infeasible = np.zeros((self.num_employees, self.num_branches))

    def step(self, action: tuple):
        employee_idx = action[0]
        branch_idx = action[1]

        is_feasible = self.is_feasible_step(employee_idx, branch_idx)
        if is_feasible:
            self.num_steps += 1
            self.update_current_state_infos(employee_idx, branch_idx)
            if self.num_steps == self.num_required_employees:
                
                self.done = True
                is_invalid = self.is_invalid_placement()
                if is_invalid:
                    self.reward = -1000
                else:
                    self.reward = self.calculate_reward()
                
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

        # TO 초과 여부
        rank = self.employees[employee_idx].rank
        if self.branches[branch_idx].current_rank_num[rank] >= self.branches[branch_idx].required_rank_num[rank]:
            return False

        # 친인척 동일지점 배치 불가

        return True

    def update_current_state_infos(self, employee_idx: int, branch_idx: int):
        """
        Update state, infeasible, branches information at every step
        """
        self.state[employee_idx][branch_idx] = 1
        self.infeasible[employee_idx, :] = 1

        employee_rank = self.employees[employee_idx].rank
        self.branches[branch_idx].current_rank_num[employee_rank] += 1
        self.branches[branch_idx].current_task_score['i'] += self.employees[employee_idx].task_score['i']
        self.branches[branch_idx].current_task_score['e'] += self.employees[employee_idx].task_score['e']


    def is_invalid_placement(self) -> bool:
        """
        Check whether placement is invalid when all steps are over
        """
        # 지점의 각 to별 정원과 실제 배치된 인원이 다를 시 해당 배치 불가능
        for i in range(len(self.branches)):
            num_required_rank = self.branches[i].required_rank_num
            num_current_rank = self.branches[i].current_rank_num
            if num_required_rank != num_current_rank : 
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

        #avg_employee_reward_by_branch = np.dot(employee_reward, self.state) / np.sum(self.state, axis=0)

        avg_employee_reward_by_branch = employee_reward

        career_score_reward = self.calculate_career_score_reward()

        branch_reward = (
            avg_employee_reward_by_branch
            + career_score_reward
        )
        reward = np.mean(branch_reward)

        return reward

    def calculate_working_months_reward(self):

        employee_working_months_reward = 0
        return employee_working_months_reward

    def calculate_preferring_branch_rotation_reward(self):
        # 직전에 선호지점에 배치되지 않았었는데, 이번에도 선호지점이 아니면 감점

        preferring_branch_rotation_reward = 0
        for employee_idx in range(self.num_employees):
            for branch_idx in range(self.num_branches):
                if (self.state[employee_idx][branch_idx] == 1 
                and not branch_idx in self.employees[employee_idx].preferred_branch_list
                and not self.employees[employee_idx].worked_at_preferred):
                    preferring_branch_rotation_reward -= 1
        

        return preferring_branch_rotation_reward

    def calculate_popular_branch_rotation_reward(self):
        # 선호지점 연속배치, 열악지점 연속배치 시 감점
        # 직전 선호지점 배치 후 이번에도 선호지점이라면 감점
        # 직전 열악지점 배치 후 이번에도 열악지점이라면 감점

        employee_popular_branch_rotation_reward = 0

        for employee_idx in range(self.num_employees):
            for branch_idx in range(self.num_branches):
                if (self.state[employee_idx][branch_idx] == 1 
                and branch_idx in self.employees[employee_idx].preferred_branch_list
                and self.employees[employee_idx].worked_at_preferred):
                    employee_popular_branch_rotation_reward -= 1

                if (self.state[employee_idx][branch_idx] == 1 
                and self.branches[branch_idx].inferior
                and self.employees[employee_idx].worked_at_inferior):
                    employee_popular_branch_rotation_reward -= 1

        return employee_popular_branch_rotation_reward

    def calculate_accessible_branch_reward(self):

        employee_accessible_branch_reward = 0
        return employee_accessible_branch_reward

    def calculate_remote_placement_priority_reward(self):

        employee_remote_placement_priority_reward = 0
        return employee_remote_placement_priority_reward

    def calculate_remote_placement_count_reward(self):
        
        remote_placement_counts_reward = 0
        return remote_placement_counts_reward

    def calculate_career_score_reward(self):
        
        career_score_reward = 0
        return career_score_reward
