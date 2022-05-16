import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from EvaluateNetwork import Build_Evaluate_Network


class Env:
    def __init__(self, model_type, reward_type, budget=10):
        self.budget = budget
        self.action_space = list(range(100))
        self.n_actions = len(self.action_space)
        self.reward_type = reward_type

        self.evaluate_network = Build_Evaluate_Network(model_type)
        print(self.evaluate_network)

        self.current_state, self.next_state = np.zeros(100), np.zeros(100)
        self.current_action = None
        self.selected_action_recoder = []
        self.available_action_space = copy.copy(self.action_space)

    def step(self, action_t):
        self.current_state = copy.copy(self.next_state)
        self.current_action = action_t
        self.next_state[self.current_action] = 1
        self.selected_action_recoder.append(action_t)
        self.available_action_space.remove(action_t)

        if len(self.selected_action_recoder) >= self.budget:
            done = True
        else:
            done = False

        if self.reward_type == "intensive":
            r = self.intensive_reward_function()
        elif self.reward_type == "global":
            if self.evaluate_final_design() >= 0:
                r = 0
            else:
                r = -self.evaluate_final_design() if done and self.evaluate_final_design() < 0 else 0
        else:
            raise Exception(
                "Please choose your reward type correctly! The type of model you are trying to build is {}!".format(self.reward_type))

        return self.current_state, self.current_action, r, self.next_state, done, self.available_action_space

    def intensive_reward_function(self):
        H_current = self.calculate_H(self.current_state)
        H_next = self.calculate_H(self.next_state)
        r = -(H_next - H_current)
        return r

    def global_reward_function(self):
        H_out = self.calculate_H(torch.FloatTensor(self.next_state.reshape(1, 100)))
        return H_out

    def calculate_H(self, state):
        H_evaluate = self.evaluate_network(torch.FloatTensor(state.reshape(1, 100)))
        return H_evaluate.squeeze().detach().numpy()

    def reset(self):
        self.current_state, self.next_state = np.zeros(100), np.zeros(100)
        self.current_action = None
        self.selected_action_recoder = []
        self.available_action_space = copy.copy(self.action_space)
        return self.current_state, self.current_action, self.next_state, self.available_action_space

    def evaluate_final_design(self):
        H_out = self.calculate_H(torch.FloatTensor(self.next_state.reshape(1, 100)))
        return H_out


if __name__ == '__main__':
    env = Env(model_type='CNN_Inception', reward_type='intensive', budget=10)
