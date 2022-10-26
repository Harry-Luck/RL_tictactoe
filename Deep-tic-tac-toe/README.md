# Tic Tac Toe Zero
Here is implemented a simplified version of Alpha Zero, reinforcement learning developed by Google Deep Mind for chess. Here it is re-purposed to play tic tac toe. Being much simpler than chess, it needs less time to train and it uses more heavily the Monte Carlo Tree search, and the neural network plays a minor role.

## Install
### Ubuntu
Needs numpy,tensorflow and progressbar to work properly. The code is currently using Tensorflow 2 and tested under Ubuntu 20.04. You can use the native pip install.

    apt-get update && apt-get install -y python3 python3-pip
    pip3 install numpy tensorflow progressbar pyyaml h5py

You can follow the instruction in the official Tensorflow website if you want to install it through Anaconda. If you have a CUDA-compatible Nvidia GPU, I suggest to install the gpu version for better performances.

    pip3 install --upgrade tensorflow-gpu
    
### Docker
    
For convinience, I also provide a Dockerfile to install all the prerequisites and a Makefile that allows to quickly build the image and open a bash in the container with a copy of the files mounted in the working volume.

    make docker
    make bash

## Structure and functionality
The algorithm implements a Monte Carlo Tree to explore the board in conjunction with a neural network. The network starts initialized at random, so it will make poor guesses on which move to take, but the value at each expolored node is updated after a terminal stated is reached (win,tie,lost). Each game is played against itself, but at each step a random action may be taken with probability *EPSILON*. After *N_ROLLOUTS* games simulated the tree is parsed to extract data and the network is trained.

The neural networks has 3 hidden layes, for a total of 99 neurons, as uses the adam optimizer. The neural networks is a policy network, thus it tries to predict the correct action from a given board of the game. The data comes directly from the monte carlo tree search, but we filter all the nodes that have not been explored enough as they lead the neural network to instability. A difference from the Alpha Zero algorithm is that in the paper the neural network has two heads, one to predict the policy and one to predict the value of the state. In my implementation I am replacing the value network with the maximum value of a policy network.

Both the neural network and the Monte Carlo tree are saved locally during the execution, so it is faster on the next execution.

## Versions
The main version is contained in the file main.py, but there are two more versions. These two versions did not achieve the same results unfortunately, but they are technically good approaches. Further developments may solve the current problems and make them achieve better performances.

The file "ttt.py" implements a DQN, similar to the one described by the Lecture n.6 in David Silver course on Reinforcement Learning (you can find it on YouTube). In the file "dd_ttt.py" several improvement have been made in the algorithm. The player is now enclosed in a class and the algorithm has been modified to implement a Double-DQN. The loss reached is equivalent or lower than the one achieved by DQN. However, the best final policy achieved still belongs to DQN. This is the reason why both version are available at the moment.  In the future, we expect the Double-DQN to achieve better results.

## Run
Download and extract the repository to your local machine. After the installation, execute the following.
To train the program instead of playing, type in your terminal

    python3 main.py train
    
To play against the algorithm, just type

    python3 main.py --play
To run the DQN algorithm, type in your terminal

    python3 ttt.py
To run the DDQN algorithm, type in your terminal

    python3 dd_ttt.py

In the last two versions, the game will ask you to play a match immediately after the training has finished and the data has been saved. During training it is possible to visualize the tensorboard. Run the following in your terminal

    tensorboard --logdir tensorflow_logs

**Disclaimer :** some of the code was written a few years ago and got deprecated. I reviewed and corrected the main part of code recently to fix that, but other parts of the code may still be broken. Run `main.py` if you want something that is sure to work.

## Results
The best result achieved for now was human-like performance after 100K episodes, in the Monte Carlo tree version. The result is achieved independently by the use of the neural network, but the purpose of the neural network will be evident in more complex games.

## External resources
https://web.stanford.edu/~surag/posts/alphazero.html

## Contact
Please write at lorenz.m97@gmail.com if you need more information or if you are interested in developing more. (cause I am!)
