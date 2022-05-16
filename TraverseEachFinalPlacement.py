import torch
from Environment import Env
from itertools import combinations
from tqdm import tqdm
import time
from EvaluateNetwork import Build_Evaluate_Network


def make_print_to_file(path='./'):
    # path， it is a path for save your log about function print
    import os
    import sys
    import datetime
    if not os.path.exists(path):
        os.makedirs(path)

    class Logger(object):
        def __init__(self, filename="enumerate.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    print("*************************************Current time is:", datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'),
          "**************************************")


def main():
    position = range(30)
    t_start = time.time()
    all_placements = list(combinations(position, 10))
    best_Hout, best_placement = 0, None
    evaluate_network = Build_Evaluate_Network('CNN_Inception')
    print(evaluate_network)
    t_mid = time.time()
    print("time_for_produce:{}".format(t_mid - t_start))
    print("Enumerate each_placement:")
    with torch.no_grad():

        for index, current_placement in enumerate(tqdm(all_placements)):
            # print('index:{},current_placement:{},\n'.format(index, current_placement))
            input_matrix = torch.zeros([1, 100])
            for item in current_placement:
                input_matrix[0, item] = 1
            input_matrix = torch.rot90(input_matrix.reshape(10, 10))
            current_Hout = evaluate_network(input_matrix)
            if current_Hout < best_Hout:
                best_Hout = current_Hout
                best_placement = input_matrix
                print("current_best_Hout:", best_Hout, "current_best_placement:\n", best_placement, "\n---------------")

    t_end = time.time()

    print("time_for_enumerate:{}".format(t_end - t_mid))

    print("best_Hout:", best_Hout)
    print("best_placement:\n", best_placement)


if __name__ == '__main__':
    make_print_to_file(path='./Log')
    main()
