# Progressive Neural Networks with PyTorch



## Description

This repository contains an implementation of Progressive Neural Networks (PNNs) in PyTorch. PNNs are designed to alleviate the problem of catastrophic forgetting in neural networks when trained on a sequence of tasks. Each new task leverages knowledge from previously learned tasks by transfer learning, while preserving their performance.





## Features

- Easy-to-use API for training and inference.

- Multi-task learning without forgetting.

- GPU support for faster computation.



## Requirements

- Python 3.x

- PyTorch (with CUDA support for GPU)

- numpy



## Installation



Clone this repository and install the required packages.



```bash
git clone https://github.com/pfagurel/ProgressiveNeuralNetwork.git
cd ProgressiveNeuralNetwork
pip install -r requirements.txt
```



## Usage



Here's a simple example of how to create a PNN and train it on two tasks.



```python

from pnn import ProgressiveNeuralNetwork

import numpy as np



# Initialize PNN

pnn = ProgressiveNeuralNetwork()



# Train on Task 1

task1_data = np.random.rand(100, 56)

task1_labels = np.random.randint(0, 2, size=100)

pnn.train_new_task(task1_data, task1_labels)



# Train on Task 2

task2_data = np.random.rand(100, 56)

task2_labels = np.random.randint(0, 2, size=100)

pnn.train_new_task(task2_data, task2_labels)



# Make predictions for Task 1

task1_preds = pnn.predict(task1_data, task_index=0)



# Make predictions for Task 2

task2_preds = pnn.predict(task2_data, task_index=1)



# Calculate accuracy for Task 1

task1_acc = pnn.accuracy(task1_data, task1_labels, task_index=0)

print(f"Task 1 accuracy: {task1_acc}%")



# Calculate accuracy for Task 2

task2_acc = pnn.accuracy(task2_data, task2_labels, task_index=1)

print(f"Task 2 accuracy: {task2_acc}%")

```




## Based on Research
This implementation is based on the research presented in the following paper, although it is not an exact replication:
> **Progressive Neural Networks**  
> *Andrei Rusu, Neil Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell*  
> Published in: *ArXiv*  
> Year: *2016*  
> [Link to the paper](https://arxiv.org/abs/1606.04671)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
