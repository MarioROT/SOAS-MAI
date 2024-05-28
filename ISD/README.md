Self-Organizing Multi-Agent Systems - Cleanup Game
This repository contains the implementation of a self-organizing multi-agent system designed to study cooperative behaviors and decision-making strategies in the Cleanup game environment that makes agents face an intertemporal social dilemma scenario.

To set up the environment and install the necessary dependencies, follow these steps:

Clone the repository:

```bash
git clone https://github.com/yourusername/cleanup-game.git
cd cleanup-game
```
Create and activate the conda environment:

Ensure you have conda installed. If not, you can install it from here.

```bash
conda env create -f environment.yml
conda activate SOAS
```
#### Training
To train the agents in the Cleanup environment, follow these steps:

Navigate to the training script:

```bash
cd src  # Adjust according to your directory structure if needed
```
Run the training script:

```bash
python training.py
```

This script initializes the environment, sets up the agents, and begins the training process. Training parameters such as the number of episodes, steps per episode, and learning rates are predefined but can be adjusted within the script.

### Monitor the training process:

Training progress will be logged to the console. When the training is finished accordig to the number of selected episodes the script will save the trained model parameters in the *trained_models* directory.

### Executing Simulations
To execute simulations using the trained agents, follow these steps:

Ensure the trained models are available:

The trained model parameters should be saved in the *trained_models* directory. If not, complete the training process first.

Run the simulation script:

``` bash
python simulate_agents.py
```

This script loads the trained models and executes simulations in the Cleanup environment. The agents' behavior will be rendered using pygame, allowing for visual observation of their actions and interactions.

### Directory Structure
The directory structure of the repository is as follows:

``` bash
IDS/
│
├── environment.yml        # Conda environment configuration file
├── README.md              # This README file
├── src/                   # Source code directory
│   ├── agent_architecture.py  # Neural network architecture for agents
│   ├── environment.py         # Custom environment definition
│   ├── simulation.py          # Simulation setup and execution
│   ├── simulation_trained.py  # Script for running simulations with trained agents
│   └── training.py        # Training script
├── trained_models/        # Directory to save trained model parameters
│   └── agent_0.pth        # Example of a saved model parameter file
└── data/                  # Directory for any dataset or additional files
```

