# D3QN-for-Optimized-Ground-Via-Placement
PyTorch implementation of the paper "Deep Reinforcement Learning-based Intelligent Design Strategy for Ground-Via Placement of System-in-Package".

Our paper is under review by IEEE Transactions on Electromagnetic Compatibility.

## Table of Contents

- [Background](#background)
- [Demo](#demo)
- [Get Started](#get-started)
- [CNN-Inception Model](#cnn-inception-model)
- [D3QN Model](#d3qn-model)
	- [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)

- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

Our goal is to find a ground-via placement strategy to achieve the best Electromagnetic Interference (EMI) mitigation for a specified number of ground vias.More details can be found in our paper.

> An example is implemented in the code, which selects 10 positions to place vias from 100 positions to minimize *H<sub>out</sub>* . 

## Demo

> Demo of placing vias yourself. 

<img src="https://github.com/ushione/D3QN-for-Optimized-Ground-Via-Placement/blob/main/demo.gif" width="320" height="200" alt="demo"/><br/>

## Get Started
Clone the project and install requirments.
> Actually if you just want to run **python** code, the basic **pytorch** package will suffice. Most packages should be version compatible. If you encounter problems, please refer to the version used by the author. 

```sh
    git clone https://github.com/ushione/D3QN-for-Optimized-Ground-Via-Placement.git
    cd D3QN-for-Optimized-Ground-Via-Placement
    pip install -r requirements.txt
```

## CNN-Inception Model
This is a CNN-Inception Model for predicting *H<sub>out</sub>* .

Folder `Data` contains Train Set `add.csv` and Test Set `test.csv`.

To train this CNN-Inception Model, run
```sh
    python TrainMyCNN_Inception.py
```

The CNN-Inception Model performance *(after 1000 epochs)* vs the number of training samples is as follows：
> |  TRAINING EXAMPLES       |     1000      |      2000      |      3000      |      4000      |
> |:------------------------:|:-------------:|:--------------:|:--------------:|:--------------:|
> | Training Loss            |     0.8078    |     0.2552     |     0.0963     |     0.0453     |
> | Testing Loss             |     7.7503    |     1.6285     |     0.6137     |     0.4015     |
> | Training Accuracy (1dB)  |     76.7%     |     94.9%      |     99.4%      |     99.9%      |
> | Testing Accuracy (1dB)   |     32.6%     |     61.9%      |     82.2%      |     88.9%      |
> | Training Accuracy (3dB)  |     99.5%     |     100%       |     100%       |     100%       |
> | Testing Accuracy (3dB)   |     72.2%     |     97.2%      |     99.7%      |     100%       |

## D3QN Model
This is a D3QN Model to optimize the ground-via placement.

### Train D3QN Model with *random sample*

> Each exploration transient *{ s<sub>t</sub> , a<sub>t</sub> , s<sub>t+1</sub> , r<sub>t</sub> }* of the agent will be stored in a replay memory.
> 
> In this paper we carefully design an ***intensive reward function*** to replace the commonly used ***global reward fuction***. (*i.e.*, *r<sub>t</sub>* will be obtained according to this intensive reward function.)
> 
> Here we will **randomly sample** transients from the replay memory to train our D3QN model.

To train this D3QN Model with ***intensive reward function***, run
```sh
    python TrainMyD3QN.py
    # or
    python TrainMyD3QN.py -r intensive
```

To train this D3QN Model with ***global reward fuction***, run
```sh
    python TrainMyD3QN.py -r global
```

> In the above code, the architecture of the D3QN model adopts the idea of CNN-Inception by default. More details can be found in our article.
> 
> If you want to modify the architecture of the D3QN model, please read and modify file `EvaluateNetwork.py` and specify the parameters *`-m`* when running `TrainMyD3QN.py`.
> 
> In other words, `python TrainMyD3QN.py` = `python TrainMyD3QN.py -m CNN_Inception`.
> 
> Here is an example of specifying a D3QN model and training it with global reward function *(The corresponding network structure and weight parameters already exist)*.

```sh
    python TrainMyD3QN.py -m DNN_Relu -r global
```

### Train D3QN Model with *Priority Experience Replay*

> Prioritized Experience Replay is a sampling method to focus on more valuable experiences and improve data utilization and training speed. In particular, it is often used in problems with sparse rewards.
> 
> *T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” arXiv preprint arXiv:1511.05952, 2015.*

If you want to use ***Priority Experience Replay*** instead of ***Random Sample*** during training the D3QN *(with global reward)*, run

```sh
    python TrainMyD3QN_PER.py -m CNN_Inception -r global
```

> In this study, we do not recommend using *Priority Experience Replay* at the same time as applying an intensive reward function, because it increases the computational burden to some extent.


