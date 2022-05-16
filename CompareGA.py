# -*- coding: utf-8 -*-
# @Python  ：python 3.6
# @Time    : 2022/3/13 15:07
# @Author  : Zheming Gu / 顾哲铭
# @Email   : guzheming@zju.edu.cn
# @File    : CompareGA.py
# @Software: PyCharm
# @Remark  : GA
# ---------------------------------------

import numpy as np
import tkinter as tk
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from geneticalgorithm import geneticalgorithm as ga
import time
import csv
from EvaluateNetwork import Build_Evaluate_Network, CNN_Inception


def f(X):
    input = torch.cuda.FloatTensor(X.reshape(1, 100))
    with torch.no_grad():
        value = net(input).cpu().numpy()[0][0]
        value = 100 if value > -37 else X.sum()
    # print(type(value))
    return value


if __name__ == '__main__':
    t0 = time.time()

    net = CNN_Inception()
    net.load_state_dict(torch.load('./Params/CNN_Inception.pkl'))
    net.cuda()

    varBound = np.array([[0, 1]] * 100)

    algorithm_param = {'max_num_iteration': 10000, \
                       'population_size': 100, \
                       'mutation_probability': 0.1, \
                       'elit_ratio': 0.01, \
                       'crossover_probability': 0.5, \
                       'parents_portion': 0.3, \
                       'crossover_type': 'uniform', \
                       'max_iteration_without_improv': None}

    model = ga(function=f, dimension=100, variable_type='int', variable_boundaries=varBound,
               algorithm_parameters=algorithm_param)

    model.run()

    allTime = time.time() - t0

    print("allTime", allTime)

    convergence = model.report
    solution = model.output_dict

    print("convergence:", type(convergence), convergence)
    print("solution:", type(solution), solution)

    allTime = time.time() - t0

    print("总时间", allTime)

    # print(np.array(convergence).reshape(10, 10))

    filename = "./GA_Result.csv"
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(convergence)