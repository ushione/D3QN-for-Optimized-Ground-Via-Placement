import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import torch.nn as nn
from Environment import Env
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F

EPISODE = 10000
BUDGET = 10
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.1
REPLAY_SIZE = 10000
BATCH_SIZE = 32
STEP = 10
GAMMA = 1
RENDER = True

cuda = True if torch.cuda.is_available() else False

"""Use cpu to accurately reproduce example results, comment out the next line if you want to use GPU."""
cuda = False

device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

seed = 888
random.seed(seed)
np.random.seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)


class D3QN_NET(nn.Module):
    def __init__(self):
        super(D3QN_NET, self).__init__()

        self.branch0 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), padding=0),
            nn.ReLU(True),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
        )

        self.branch2 = nn.Sequential(
            # input shape: batch*1*10*10
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), padding=3),
            nn.ReLU(True),
        )

        self.conv1 = nn.Conv2d(25, 12, kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12 * 10 * 10, 200)

        self.fc1 = nn.Linear(200, 200)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.advantage_fc = nn.Linear(200, 100)
        self.advantage_fc.weight.data.normal_(0, 0.1)  # initialization
        self.value_fc = torch.nn.Linear(200, 1)
        self.value_fc.weight.data.normal_(0, 0.2)

    def _forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = torch.cat((branch0, branch1, branch2, branch3), dim=1)
        return outputs

    def forward(self, x):
        x = x.reshape(-1, 1, 10, 10)
        output = self._forward(x)
        output = F.relu(self.conv1(output))
        output = self.flatten(output)
        output = F.relu(self.linear1(output))

        x = F.relu(self.fc1(output))
        a_value = self.advantage_fc(x)
        v_value = self.value_fc(x)
        Q_value = v_value + (a_value - torch.mean(a_value, axis=-1, keepdim=True))
        return Q_value


class Memory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return np.array(random.sample(self.memory, n))

    def __len__(self):
        return len(self.memory)


class D3QN:
    def __init__(self):
        self.evaluate_Q_net, self.target_Q_net = D3QN_NET().to(device), D3QN_NET().to(device)
        self.epsilon = INITIAL_EPSILON
        self.replay_total, self.time_step = 0, 0
        self.memory = Memory(capacity=REPLAY_SIZE)
        self.optimizer = torch.optim.Adam(params=self.evaluate_Q_net.parameters(), lr=0.001)
        self.loss_func = torch.nn.MSELoss()

    def choose_action(self, current_state, available_action_space):
        if np.random.uniform() < self.epsilon:  # random
            action = random.choice(available_action_space)

        else:  # greedy
            x = torch.unsqueeze(Tensor(current_state), 0)
            with torch.no_grad():
                Q_value = self.evaluate_Q_net.forward(x)
                check_point = torch.sort(Q_value, 1, descending=True)[1].squeeze().tolist()
                action_order = [i for i in torch.sort(Q_value, 1, descending=True)[1].squeeze().tolist() if
                                i in available_action_space]
                action = action_order[0]

        self.epsilon = max(self.epsilon - 0.001, 0.1)
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))
        self.memory.store(transition)
        if len(self.memory) >= BATCH_SIZE:
            self.train_my_D3QN()

    def train_my_D3QN(self):
        batch_sample = self.memory.sample(BATCH_SIZE)
        batch_s = Tensor(batch_sample[:, 0:100])
        batch_a = LongTensor(batch_sample[:, 100:101])
        batch_r = Tensor(batch_sample[:, 101:102])
        batch_s_ = Tensor(batch_sample[:, 102:202])

        evaluate_max_action = torch.argmax(self.evaluate_Q_net(batch_s_), dim=1)
        target_update_Q = self.target_Q_net(batch_s_).detach()

        batch_target = Tensor(BATCH_SIZE, 1)
        for i in range(BATCH_SIZE):
            done = batch_sample[i, 202]
            if done:
                batch_target[i] = batch_r[i]
            else:
                batch_target[i] = batch_r[i] + GAMMA * target_update_Q[i, evaluate_max_action[i]]

        batch_prediction = self.evaluate_Q_net(batch_s).gather(1, batch_a)

        loss = self.loss_func(batch_prediction, batch_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_params(self):
        torch.save(self.evaluate_Q_net.state_dict(), './Params/D3QN_params.pkl')
        self.target_Q_net.load_state_dict(torch.load('./Params/D3QN_params.pkl'))


def main():
    env = Env(model_type=model_type, reward_type=reward_type, budget=BUDGET)
    RL = D3QN()

    # plt.style.use('seaborn')
    figure, ax = plt.subplots(1, 2, figsize=(10, 4))
    plt.ion()
    recoder = []
    t_recoder = []
    t0 = time()

    for episode in tqdm(range(EPISODE)):
        s, a, s_, available_action_space = env.reset()
        for step in range(BUDGET):
            action = RL.choose_action(s, available_action_space)
            s, a, r, s_, done, available_action_space = env.step(action)
            RL.store_transition(s, a, r, s_, done)
            s = s_

            if episode % 1000 == 0:
                RL.update_target_params()

            if done:
                assert step == BUDGET - 1
                Hout_of_this_episode = env.evaluate_final_design()
                recoder.append(Hout_of_this_episode)

                if RENDER:
                    final_placement = np.zeros(100)
                    for i in available_action_space:
                        final_placement[i] = 1

                    ax[0].cla()
                    ax[0].imshow(final_placement.reshape((10, 10)))

                    ax[1].cla()
                    ax[1].plot(range(len(recoder)), recoder)
                    ax[1].set_title("Episode = {},Random_rate = {}".format(episode, round(RL.epsilon, 4)))

                    plt.show()
                    plt.pause(0.01)

                if recoder != [] and Hout_of_this_episode <= np.min(recoder):
                    print(Hout_of_this_episode, [i for i in range(100) if i not in available_action_space])
                    print(env.next_state.reshape(10, 10))
                    plt.savefig("current_optimal_placement.jpg")

                t_recoder.append(time() - t0)
                df = pd.DataFrame(np.array([recoder, t_recoder]).T)
                df.to_csv("D3QN_Result.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UI of placing vias yourself.')
    parser.add_argument('-m', '--model_type', type=str, default='CNN_Inception',
                        help='Please read EvaluateNetwork.py before choose your model.')
    parser.add_argument('-r', '--reward_type', type=str, default='intensive', help='Please select the type of reward.')
    args = parser.parse_args()
    model_type = args.model_type
    reward_type = args.reward_type
    main()
