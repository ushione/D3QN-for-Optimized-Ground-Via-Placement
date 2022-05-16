# D3QN-for-Optimized-Ground-Via-Placement
![python_verion](https://user-images.githubusercontent.com/87009163/168636733-f6c303bd-6f3f-4215-b008-c37ea05c0921.svg)

PyTorch implementation of the paper "Deep Reinforcement Learning-based Intelligent Design Strategy for Ground-Via Placement of System-in-Package".

Our paper is under review by IEEE Transactions on Electromagnetic Compatibility.

## Table of Contents

- [Background](#background)
- [Demo](#demo)
- [Get Started](#get-started)
- [CNN-Inception Model](#cnn-inception-model)
- [D3QN Model](#d3qn-model)
	- [Train D3QN Model with *random sample*](#train-d3qn-model-with-random-sample)
	- [Train D3QN Model with *Priority Experience Replay*](#train-d3qn-model-with-priority-experience-replay)
- [Genetic Algorithm](#genetic-algorithm)
- [UI for manually optimizing ground-via placement](#ui-for-manually-optimizing-ground-via-placement)
- [Validity Verification](#validity-verification)

## Background

Our goal is to find a ground-via placement strategy to achieve the best Electromagnetic Interference (EMI) mitigation for a specified number of ground vias.More details can be found in our paper.

> An example is implemented in the code, which selects 10 positions to place vias from 100 positions to minimize *H<sub>out</sub>* . 

## Demo

> Demo of placing vias yourself. 

<img id="demo_gif" src="https://github.com/ushione/D3QN-for-Optimized-Ground-Via-Placement/blob/main/demo.gif" width="320" height="200" alt="demo"/><br/>

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

The CNN-Inception Model performance *(after 1000 epochs)* vs the number of training samples is as follows：  <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>
> |  TRAINING EXAMPLES       |     1000      |      2000      |      3000      |      4000      |
> |:------------------------:|:-------------:|:--------------:|:--------------:|:--------------:|
> | Training Loss            |     0.8078    |     0.2552     |     0.0963     |     0.0453     |
> | Testing Loss             |     7.7503    |     1.6285     |     0.6137     |     0.4015     |
> | Training Accuracy (1dB)  |     76.7%     |     94.9%      |     99.4%      |     99.9%      |
> | Testing Accuracy (1dB)   |     32.6%     |     61.9%      |     82.2%      |     88.9%      |
> | Training Accuracy (3dB)  |     99.5%     |     100%       |     100%       |     100%       |
> | Testing Accuracy (3dB)   |     72.2%     |     97.2%      |     99.7%      |     100%       |

## D3QN Model
This is a D3QN Model to optimize the ground-via placement.  <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>

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

> Here is the result of the D3QN *(with intensive reward function)* training and learning (note that the algorithm learns and explores the optimal placement).
> 
<img id="optimal_placement" src="https://github.com/ushione/D3QN-for-Optimized-Ground-Via-Placement/blob/main/current_optimal_placement.jpg" width="500" height="200" alt="current_optimal_placement"/><br/>

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
> Here is an example of specifying a D3QN model and training it with global reward function *(The corresponding network structure and weight parameters already exist)*.  <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>

```sh
    python TrainMyD3QN.py -m DNN_Relu -r global
```

### Train D3QN Model with *Priority Experience Replay*

> Prioritized Experience Replay is a sampling method to focus on more valuable experiences and improve data utilization and training speed. In particular, it is often used in problems with sparse rewards.
> 
> *Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint [arXiv:1511.05952 (2015)](https://arxiv.org/abs/1511.05952).*

If you want to use ***Priority Experience Replay*** instead of ***Random Sample*** during training the D3QN *(with global reward)*, run

```sh
    python TrainMyD3QN_PER.py -m CNN_Inception -r global
```

> In this study, we do not recommend using *Priority Experience Replay* at the same time as applying an intensive reward function, because it increases the computational burden to some extent. <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>

## Genetic Algorithm

We also try to solve an approximation problem using the ***Genetic Algorithm*** (with some tweaks to facilitate GA implementation). <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>

```sh
    python CompareGA.py
```

## UI for manually optimizing ground-via placement

If you want to try to manually optimize the ground-via placement, we provide a UI panel.

> After you have pyqt5 installed, run
```sh
    python UI.py
```

Now you can try placing the ground vias manually like in the [demo](#demo_gif). Try to adjust your scheme to minimize the calculation result! <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>

## Validity Verification

In addition to the verification of the algorithm in our paper, we try to traverse every possible solution to find the optimal one. 
> (Accurately, this is a traversal based on prior knowledge. We know that in the case of fewer vias, ground vias should be placed near the radiation source. Therefore, we only consider candidate positions close to the radiation source (left Three columns). The number of solutions that need to be traversed is 
*`C(30, 10) = 30045015`*)
> 
> You can run `python TraverseEachFinalPlacement.py`, but this may require you to have a GPU. Also it will take a long time to run, so we recommend looking directly at our result log `Log/Enumerate_log.txt` if you are interested. In fact, [this image](#optimal_placement) should be the best solution.
> 
> Although due to the use of a neural network to asymptotically act on the value function ***Q<sup>\*</sup>(s<sub>t</sub> , a<sub>t</sub>)***, we cannot guarantee that D3QN will converge to such an optimal solution every time. But it makes sense to demonstrate the effectiveness of our algorithm. <span style='font-size:5px;'>&nbsp;&nbsp;&nbsp;&nbsp;[[Back to Top]](#d3qn-for-optimized-ground-via-placement)</span>
