# MazeRunner

A maze solver to find the shortest path from start to end using Deep Q Learning

## Getting Started

This project is console only at the moment. I may add UI later. 

### Prerequisites

The project needs certain python libraries that might require you to have either GPU support on your PC, or an AVX enabled CPU. This is solely for the training of Neural Network.

*Make sure you have Python, pip and Git installed before proceeding any further*

### Installing

To get the project running we'll need to set up a virtual environment. Search the internet for how you'd do that if you don't already know.
To get the dependency packages, run the following commands in a command prompt or terminal
```
pip install numpy
pip install matplotlib
pip install tensorflow
pip install keras
```

Once the installation is finished

You can run the main.py file in the project directory :
```
python main.py
```
Wait for the training to complete
![](https://user-images.githubusercontent.com/47733983/79787474-aebfe000-8364-11ea-97ed-b7ef8f7eb392.png)

You should see a solved maze with the path at the end
![](https://user-images.githubusercontent.com/47733983/79787557-ceef9f00-8364-11ea-8b6b-4ad1840b0754.png)

## Changing the maze

The project is not yet complete so you'd have to do this manually bby modifying the main.py file.
You should look for a numpy array looking named as maze, which you need to modify. Ones refer to the path accessible while zeros refer to blocked cells.
![](https://user-images.githubusercontent.com/47733983/79787828-3dccf800-8365-11ea-9bda-ebae3e9b1558.png)


## Built With

* [Keras](https://keras.io/) - Used for training the neural network model


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Kudos to [Samy Zafrany](https://github.com/samyzaf), whose code I took inspiration from

