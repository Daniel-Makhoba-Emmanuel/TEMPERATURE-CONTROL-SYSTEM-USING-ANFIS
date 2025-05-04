# Adaptive Fuzzy Logic Control for a Temperature System

## Description

This project focuses on the design, implementation, and evaluation of an adaptive fuzzy logic controller for a simulated first-order temperature system. The goal was to explore data-driven control methods by training a controller to mimic the behavior of a well-tuned conventional PID controller and comparing their performance under various simulated conditions.

The project involves modeling the temperature system, generating training data using a PID controller, designing and training an ANFIS-like fuzzy controller using a custom NumPy implementation and optimization, simulating both controllers, and comparing their performance using plots and quantitative metrics.

## Methodology

1. System Modeling: A first-order linear ordinary differential equation was used to model the temperature system dynamics.

2. PID Control: A conventional PID controller was implemented and tuned to provide effective temperature control for the simulated system.

3. Training Data Generation: The tuned PID controller was used to simulate the system under varying setpoints and ambient temperatures, recording the error, change in error, and PID output power as training data.

4. Adaptive Fuzzy Controller Design: A zero-order Sugeno-type fuzzy inference system with 5 triangular membership functions per input (Error, Delta Error) and 25 singleton consequents was designed.

5. Controller Training: A custom NumPy-based implementation of the Sugeno inference process was developed. The controller's parameters (MF shapes/positions and singleton values) were optimized using `scipy.optimize.least_squares` to minimize the error between the fuzzy controller's output and the target PID output from the training data.

6. Simulation & Comparison: Both the trained adaptive fuzzy controller and the tuned PID controller were simulated on the identical temperature system model and test conditions. Performance was compared visually using plots and quantitatively using the Root Mean Squared Error (RMSE) of the temperature error.


## RESULTS

The training process successfully yielded a set of parameters for the adaptive fuzzy controller that allowed it to approximate the PID controller's output on the training data.

Simulation results showed that the trained adaptive fuzzy controller successfully mimicked the general behavior of the PID controller in terms of setpoint tracking and handling disturbances. Visual comparison of temperature response and controller output plots indicated similar performance characteristics.

The Temperature Error Comparison plot highlighted subtle differences, with the PID controller generally achieving slightly smaller errors during steady-state periods. This was supported by the quantitative RMSE metric, where the PID controller demonstrated a lower overall RMSE compared to the adaptive fuzzy controller in the simulated scenario.

## Code Files
- `generate_anfis_training_data.py`: Simulates the PID-controlled temperature system and generates the training data CSV.

- `train_anfis_controller.py`: Implements the NumPy-based Sugeno fuzzy inference and trains the controller parameters using `scipy.optimize.least_squares`. Saves the trained parameters to a JSON file.

- `run_simulations.py`: Loads the trained fuzzy controller and simulates both the ANFIS and PID controllers on the temperature system. Saves simulation results to CSV files.

- `compare_results.py`: Loads the simulation results from the CSV files, generates comparison plots, and calculates performance metrics (RMSE).

## Requirements
- Python 3.x

- NumPy

- SciPy

- Pandas

- Matplotlib

## Setup and Usage
- Clone this repository:


- Install the requirements

- Generate training data:

`python generate_anfis_training_data.py`

This will create anfis_training_data_pid.csv. You may need to close the plot window for the script to finish.

- Train the adaptive fuzzy controller:

`python train_anfis_controller.py`

This will train the controller and save parameters to `trained_anfis_controller_info_numpy.json`. You may need to close the plot window for the script to finish.

- Run simulations for both controllers:

`python run_simulations.py`

This will create `anfis_simulation_results.csv` and `pid_simulation_results.csv`.

- Compare the simulation results:

`python compare_results.py`

This will display the comparison plots and print the RMSE values.