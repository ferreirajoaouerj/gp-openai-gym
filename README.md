# Genetic Programming scripts for OpenAI Gym environments

Python scripts implementing genetic programming (in DEAP) for some OpenAI Gym environments. Includes basic python scripts for training and visualizing DQN and DDPG reinforcement learning agents, using Stable-Baselines, for comparison.

* [DEAP](https://github.com/deap/deap) - Distributed Evolutionary Algorithms for Python
* [OpenAI Gym](https://gym.openai.com/) - Toolkit for developing and comparing reinforcement learning algorithms
* [Pybullet Gym](https://github.com/benelot/pybullet-gym) - Open-source implementations of OpenAI Gym MuJoCo environments for use with the OpenAI Gym 
* [Stable-Baselines](https://github.com/hill-a/stable-baselines) - A fork of OpenAI Baselines, implementations of reinforcement learning algorithms 


## Getting Started

Clone the repository and run the script corresponding to the Gym environment.

You will be asked if you want to start a new run or load statistics from another run. In case you start a new run, you will have the option to save the statistics (including the hall of the fame objects, containing the best individuals) and overwrite a possibly existing one.

In case you save the statistics or decide to not run again, the pickle file will be loaded (if it exists in the same directory) and statistics will be shown.

Most parameters can be edited in the initial lines by changing the dictionary values.

A new environment is implemented using 'Gym' interface named 'Carrinho', and is present in the 'gym-carinho' folder. It deals with the motion planning of a car-like robot.

### Prerequisites

Please refer to DEAP, Gym, Pybullet-Gym and Stable-Baselines documentation for detailed instructions.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks for the people involved in the DEAP project
* Thanks for everyone contributing to OpenAI Gym and Pybullet-Gym
* Thanks araffin for the stable-baselines package

