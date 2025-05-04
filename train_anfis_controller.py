import pandas as pd
import numpy as np
from scipy.optimize import least_squares # Our optimization tool
import matplotlib.pyplot as plt
import json # To save/load controller info
import sys # To exit if data loading fails

# --- Step 3.2a: Load and Prepare Data ---
# Load the training data generated in Step 3.1 (generate_anfis_training_data.py)
csv_filename_anfis = 'anfis_training_data_pid.csv'
try:
    df_anfis_training = pd.read_csv(csv_filename_anfis)
    print(f"Successfully loaded data from {csv_filename_anfis}")
except FileNotFoundError:
    print(f"Error: {csv_filename_anfis} not found. Make sure you have run the Step 3.1 script.")
    sys.exit(f"Required data file not found: {csv_filename_anfis}") # Exit if the file is not found
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    sys.exit(f"Failed to load data from {csv_filename_anfis}")


# --- Prepare Input and Output Data for Training ---
# These are the inputs and target output for our adaptive fuzzy controller.
# The controller will learn to map (Error, Delta_Error) -> PID_Output_Power
errors = df_anfis_training['Error'].values
delta_errors = df_anfis_training['Delta_Error'].values
target_powers = df_anfis_training['PID_Output_Power'].values

# --- Analyze Data Ranges (Essential for defining fuzzy universes and initial MFs) ---
# Determine the range of values for Error and Delta_Error from the training data.
# This helps in defining appropriate universes of discourse for the fuzzy variables.
error_range = (np.min(errors), np.max(errors))
delta_error_range = (np.min(delta_errors), np.max(delta_errors))
power_range = (0, 1) # Output power is always between 0 and 1

print(f"Error range from data: {error_range}")
print(f"Delta Error range from data: {delta_error_range}")
print(f"Number of training data points: {len(errors)}")

# --- Step 3.2b: Define Fuzzy System Structure (Number of MFs, Type, etc.) ---
# We define the structure and initial parameters conceptually here.
# The parameters themselves will be stored and manipulated in a flattened array.

num_mfs_error = 5      # Number of membership functions for Error
num_mfs_delta_error = 5 # Number of membership functions for Delta Error
mf_type = 'trimf'      # Membership function type (using triangular MFs)
and_method = np.fmin   # Fuzzy AND operator (min)
# and_method = np.multiply # Fuzzy AND operator (product - another common choice)

# Total number of output singletons = number of rules (grid partition)
num_singletons = num_mfs_error * num_mfs_delta_error


# --- Define Membership Function Implementations (NumPy based) ---
# Implement the chosen membership function type using NumPy.
# This function will be used within our custom inference logic.

def triangular_mf(x, params):
    """Triangular membership function (a, b, c)."""
    # Ensure x is a NumPy array for element-wise operations
    x = np.asarray(x)
    a, b, c = params
    # Handle edge cases where triangle parameters are invalid (e.g., a >= b or b >= c)
    if a > b or b > c:
         # Return zero membership for all inputs if parameters are invalid
         return np.zeros_like(x, dtype=np.float64)
    if a == b and b == c: # Handle a single point MF (a, a, a)
         y = np.zeros_like(x, dtype=np.float64)
         y[x == a] = 1.0
         return y


    y = np.zeros_like(x, dtype=np.float64)

    # Left slope: a to b
    if b != a:
        mask_left = np.logical_and(x >= a, x <= b)
        y[mask_left] = (x[mask_left] - a) / (b - a)

    # Right slope: b to c
    if c != b:
        mask_right = np.logical_and(x >= b, x <= c)
        y[mask_right] = (c - x[mask_right]) / (c - b)

    # Peak: at b
    y[x == b] = 1.0

    # Ensure membership is between 0 and 1 (should be handled by the logic above, but as a safeguard)
    y = np.clip(y, 0.0, 1.0)

    return y

# Choose the MF implementation based on mf_type
if mf_type == 'trimf':
    mf_function = triangular_mf
else:
    raise ValueError(f"Unsupported membership function type: {mf_type}")


# --- Step 3.2c: Define Initial Parameters for Optimization ---
# These parameters define the initial shape of the MFs and the initial singleton values.
# Order: [Error MF1 params, Error MF2 params, ..., Delta Error MF1 params, ..., Output Singleton 1 value, ...]

initial_parameters = []

# Initial parameters for Input Triangular MFs ([a, b, c])
# Space the initial centers evenly across the data ranges and set widths.
# Using data ranges instead of universes directly, as our custom logic doesn't strictly need universes defined beforehand,
# but universes are good for conceptualizing the MF placement.
error_centers = np.linspace(error_range[0], error_range[1], num_mfs_error)
# A simple heuristic for initial width. You can adjust this heuristic.
error_mf_width = (error_range[1] - error_range[0]) / (num_mfs_error * 1.5)

initial_error_mf_params = [] # List of [a, b, c] for each error MF
for i in range(num_mfs_error):
    center = error_centers[i]
    a = center - error_mf_width
    b = center
    c = center + error_mf_width
    initial_error_mf_params.append([a, b, c])
    initial_parameters.extend([a, b, c]) # Add to the flattened parameter list

delta_error_centers = np.linspace(delta_error_range[0], delta_error_range[1], num_mfs_delta_error)
delta_error_mf_width = (delta_error_range[1] - delta_error_range[0]) / (num_mfs_delta_error * 1.5)

initial_delta_error_mf_params = [] # List of [a, b, c] for each delta error MF
for i in range(num_mfs_delta_error):
    center = delta_error_centers[i]
    a = center - delta_error_mf_width
    b = center
    c = center + delta_error_mf_width
    initial_delta_error_mf_params.append([a, b, c])
    initial_parameters.extend([a, b, c]) # Add to the flattened parameter list


# Initial values for Output Singletons
# Spread initial values evenly across the output power range [0, 1]
initial_singleton_values = np.linspace(0, 1, num_singletons)
initial_parameters.extend(initial_singleton_values.tolist()) # Add to the flattened parameter list

initial_parameters = np.array(initial_parameters)
print(f"Total number of parameters to optimize: {len(initial_parameters)}")


# --- Step 3.2d: Implement Custom Sugeno Fuzzy Inference (NumPy) ---
# This function takes the input data and the current parameters and performs fuzzy inference.

def predict_sugeno(params, errors, delta_errors, num_mfs_error, num_mfs_delta_error, mf_function, and_method):
    """
    Performs Sugeno (Type 0) fuzzy inference for all data points using given parameters.
    Returns the predicted output power for each input pair.
    """
    num_data_points = len(errors)
    predicted_powers = np.zeros(num_data_points)

    # Unpack parameters: Input MF params followed by Output Singleton values
    # Number of parameters per input MF depends on the MF type (e.g., 3 for trimf)
    # This logic needs to match the order and number of parameters added in initial_parameters.
    params_per_mf = 3 # Assuming 'trimf' as defined by mf_type

    num_input_mf_params = (num_mfs_error + num_mfs_delta_error) * params_per_mf
    expected_total_params = num_input_mf_params + num_singletons

    if len(params) != expected_total_params:
        # This check helps catch errors if the parameter collection logic doesn't match the unpacking logic.
        print(f"Parameter count mismatch in predict_sugeno. Expected {expected_total_params}, got {len(params)}")
        # Return NaNs or raise an error if parameter count is fundamentally wrong.
        return np.full(num_data_points, np.nan)


    # Extract input MF parameters from the flattened array
    error_mf_params_flat = params[:num_mfs_error * params_per_mf]
    delta_error_mf_params_flat = params[num_mfs_error * params_per_mf : num_input_mf_params]

    # Extract output singleton values from the end of the flattened array
    singleton_values = params[num_input_mf_params:]

    # Reshape input MF parameters for easier access [num_mfs, params_per_mf]
    error_mf_params = error_mf_params_flat.reshape(num_mfs_error, params_per_mf)
    delta_error_mf_params = delta_error_mf_params_flat.reshape(num_mfs_delta_error, params_per_mf)


    # --- Perform inference for each data point ---
    for i in range(num_data_points):
        input_error = errors[i]
        input_delta_error = delta_errors[i]

        # Fuzzification: Calculate membership degrees for all MFs for the current inputs
        # Use the mf_function (e.g., triangular_mf) with the current input value and MF parameters.
        # Resulting arrays: [num_mfs_error], [num_mfs_delta_error]
        error_memberships = np.array([mf_function(np.array([input_error]), mf_params)[0] for mf_params in error_mf_params])
        delta_error_memberships = np.array([mf_function(np.array([input_delta_error]), mf_params)[0] for mf_params in delta_error_mf_params])

        # Rule Firing Strength (Antecedent): Apply fuzzy AND operator
        # Grid partition: combine every error MF membership with every delta error MF membership.
        # The result is the firing strength of each rule.
        firing_strengths = np.zeros(num_singletons)
        rule_index = 0
        for j in range(num_mfs_error):
            for k in range(num_mfs_delta_error):
                # Apply the chosen AND method (e.g., min or product) to the membership degrees.
                firing_strengths[rule_index] = and_method(error_memberships[j], delta_error_memberships[k])
                rule_index += 1

        # Normalization (Optional but common in ANFIS)
        sum_firing_strengths = np.sum(firing_strengths)

        if sum_firing_strengths == 0 or np.isnan(sum_firing_strengths):
            # If no rules fire (sum of firing strengths is zero) or if it's NaN (due to invalid MF params),
            # return a default value (e.g., average of singleton values or 0.5, which is mid-range power).
            # This helps the optimizer handle potentially invalid parameter sets.
            predicted_power = np.mean(singleton_values) if singleton_values.size > 0 and not np.isnan(np.mean(singleton_values)) else 0.5
        else:
            # Weighted Average (Defuzzification for Sugeno-0): sum(firing_strength * singleton_value) / sum(firing_strength)
            # Weights are the firing strengths, values are the singleton values.
            predicted_power = np.sum(firing_strengths * singleton_values) / sum_firing_strengths

        # Clamp the output to the valid physical range [0, 1] for heater power.
        predicted_powers[i] = max(0.0, min(1.0, predicted_power))

    return predicted_powers


# --- Step 3.2e: Define the Error Function for Optimization ---
# This function is what scipy.optimize.least_squares will try to minimize.
# It takes the current parameters from the optimizer, calculates the fuzzy system's
# output for all training data, and returns the array of errors.

def sugeno_error(params, errors, delta_errors, target_powers, num_mfs_error, num_mfs_delta_error, mf_function, and_method):
    """
    Evaluates the Sugeno fuzzy system with the given parameters for all training data points.
    Returns the array of errors (predicted_output - target_output).
    This is the function that scipy.optimize.least_squares minimizes the sum of squares of.
    """
    # Get predicted outputs using the current parameters by calling the inference function.
    predicted_powers = predict_sugeno(
        params, errors, delta_errors,
        num_mfs_error, num_mfs_delta_error,
        mf_function, and_method
    )

    # Check if predict_sugeno returned NaNs (indicating an issue with parameters)
    if np.any(np.isnan(predicted_powers)):
         print("Warning: predict_sugeno returned NaN values.")
         # Return a large error if prediction failed for some data points
         return np.full(len(errors), np.inf)


    # Calculate the error array (difference between predicted and target outputs)
    errors_array = predicted_powers - target_powers

    return errors_array # least_squares minimizes the sum of squares of this array


# --- Step 3.2f: Implement Parameter Optimization ---

print("Starting parameter optimization (Manual NumPy Implementation)...")
print(f"Optimizing {len(initial_parameters)} parameters...")

# Use least_squares from scipy.optimize to find the parameters that minimize the fuzzy_system_error.
# This is where the learning happens.
# least_squares minimizes the sum of squares of the array returned by fuzzy_system_error.
optimization_result = least_squares(
    sugeno_error,              # The function to minimize (our custom error function)
    initial_parameters,        # Initial guess for parameters
    # args is a tuple of additional arguments passed to sugeno_error after 'params'.
    args=(errors, delta_errors, target_powers, num_mfs_error, num_mfs_delta_error, mf_function, and_method),
    verbose=2,                 # Set to 1 or 2 to see optimization progress (shows iterations, cost, etc.)
    # bounds=(-np.inf, np.inf), # Optional: Add bounds if needed (e.g., MF parameters ordering, singleton values 0-1)
    # method='trf', # Trust Region Reflective algorithm often works well with bounds. Default 'lm' (Levenberg-Marquardt) is also common.
    max_nfev=5000 # Increased max function evaluations - training can take many iterations for good convergence. Adjust if needed based on training time vs. performance improvement.
)

optimized_parameters = optimization_result.x

print("Optimization complete.")
print(f"Optimization success: {optimization_result.success}")
print(f"Optimization message: {optimization_result.message}")
print(f"Final cost (sum of squared errors): {optimization_result.cost}")
# Calculate RMSE from the final cost. RMSE = sqrt(mean(errors^2)) = sqrt(sum(errors^2) / N) = sqrt(cost / N)
final_rmse = np.sqrt(optimization_result.cost / len(errors))
print(f"Final RMSE after optimization: {final_rmse:.4f}")


# --- Step 3.2g: Evaluate and Save the Trained System ---
# Evaluate the performance of the trained parameters on the training data.

predicted_powers_trained = predict_sugeno(
    optimized_parameters, errors, delta_errors,
    num_mfs_error, num_mfs_delta_error,
    mf_function, and_method
)

# Check if predict_sugeno returned NaNs after optimization
if np.any(np.isnan(predicted_powers_trained)):
     print("Warning: predict_sugeno returned NaN values for trained parameters.")
     # Handle this case - maybe the trained parameters are invalid.
     # For now, proceed but be aware of potential issues.


# Calculate RMSE on the training data (should closely match the final_rmse from optimization)
rmse_check = np.sqrt(np.mean((predicted_powers_trained - target_powers)**2))
print(f"RMSE check on training data using trained parameters: {rmse_check:.4f}")


# --- Plot Actual vs. Predicted Outputs ---
# Visualize how well the trained controller's output matches the target PID output on the training data.
plt.figure(figsize=(8, 6))
plt.scatter(target_powers, predicted_powers_trained, alpha=0.5, label='Predicted vs. Target')
plt.xlabel("Target PID Output Power")
plt.ylabel("Predicted Adaptive Fuzzy Output Power")
plt.title("Trained Adaptive Fuzzy Controller Prediction vs. Target PID Output (NumPy)")
plt.grid(True)
plt.axis('equal') # Make axes equal scale for a better visual comparison to the ideal line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Ideal Prediction') # Plot y=x line for reference
plt.xlim(-0.1, 1.1) # Add small buffer to limits for better visualization
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()


# --- Save the Trained Controller Information ---
# We save the optimized parameters and the structure information needed to use the controller later.
# This allows us to load the trained controller without retraining every time.

trained_controller_info = {
    'optimized_parameters': optimized_parameters.tolist(), # Save the learned parameters (convert numpy array to list for JSON)
    'num_mfs_error': num_mfs_error,
    'num_mfs_delta_error': num_mfs_delta_error,
    'mf_type': mf_type, # Save the MF type used (e.g., 'trimf')
    # Save the AND method used (e.g., 'min' or 'product')
    'and_method': 'min' if and_method == np.fmin else ('product' if and_method == np.multiply else str(and_method)),
    # Saving ranges/universes used for initialisation can also be helpful for deployment,
    # especially if you need to clamp inputs in the deployment code.
    'error_range_used_for_init': error_range,
    'delta_error_range_used_for_init': delta_error_range,
    'power_range': power_range, # The output range [0, 1]
}

# Define the filename for the JSON file.
params_filename = 'trained_anfis_controller_info_numpy.json'
print(f"Attempting to save trained controller information to {params_filename}...") # Debug print

# Save the dictionary to a JSON file.
try:
    with open(params_filename, 'w') as f:
        json.dump(trained_controller_info, f, indent=4)
    print(f"Trained controller information successfully saved to {params_filename}")
except IOError as e:
    print(f"Error saving trained controller information to {params_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred while saving the controller information: {e}")


# --- How to load parameters and use the controller later for simulation (Step 4) ---
"""
# This commented-out section shows the logic you would use in Step 4 to load
# the saved controller information and recreate the fuzzy inference logic for deployment.

import json
import numpy as np

# Need to define the same membership function implementation (e.g., triangular_mf)
def triangular_mf(x, params):
    # Ensure x is a NumPy array for element-wise operations
    x = np.asarray(x)
    a, b, c = params
    if a > b or b > c: # Handle invalid parameters
        return np.zeros_like(x, dtype=np.float64)
    if a == b and b == c: # Handle a single point MF
         y = np.zeros_like(x, dtype=np.float64)
         y[x == a] = 1.0
         return y

    y = np.zeros_like(x, dtype=np.float64)
    if b != a:
        mask_left = np.logical_and(x >= a, x <= b)
        y[mask_left] = (x[mask_left] - a) / (b - a)
    if c != b:
        mask_right = np.logical_and(x >= b, x <= c)
        y[mask_right] = (c - x[mask_right]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


# Need to map the saved 'and_method' string back to the NumPy function
def get_and_method_func(method_str):
    if method_str == 'min':
        return np.fmin
    elif method_str == 'product':
        return np.multiply
    else:
        raise ValueError(f"Unsupported AND method string: {method_str}")


def use_trained_sugeno(input_error, input_delta_error, trained_params, num_mfs_error, num_mfs_delta_error, mf_function, and_method_func, error_range_init, delta_error_range_init, power_range_init):
"""
#Uses the trained Sugeno controller (NumPy implementation) to predict output for single inputs.
#Includes clamping inputs to the ranges used during training.
"""
    # Clamp inputs to the ranges used during training for consistency
    input_error_clamped = max(error_range_init[0], min(error_range_init[1], input_error))
    input_delta_error_clamped = max(delta_error_range_init[0], min(delta_error_range_init[1], input_delta_error))


    params = trained_params
    params_per_mf = 3 # Assuming 'trimf'

    num_input_mf_params = (num_mfs_error + num_mfs_delta_error) * params_per_mf
    num_singletons = num_mfs_error * num_mfs_delta_error

    # Unpack parameters
    error_mf_params_flat = params[:num_mfs_error * params_per_mf]
    delta_error_mf_params_flat = params[num_mfs_error * params_per_mf : num_input_mf_params]
    singleton_values = params[num_input_mf_params:]

    error_mf_params = error_mf_params_flat.reshape(num_mfs_error, params_per_mf)
    delta_error_mf_params = delta_error_mf_params_flat.reshape(num_mfs_delta_error, params_per_mf)


    # Fuzzification
    error_memberships = np.array([mf_function(np.array([input_error_clamped]), mf_params)[0] for mf_params in error_mf_params])
    delta_error_memberships = np.array([mf_function(np.array([input_delta_error_clamped]), mf_params)[0] for mf_params in delta_error_mf_params])

    # Rule Firing Strength (Grid Partition)
    firing_strengths = np.zeros(num_singletons)
    rule_index = 0
    for j in range(num_mfs_error):
        for k in range(num_mfs_delta_error):
            firing_strengths[rule_index] = and_method_func(error_memberships[j], delta_error_memberships[k])
            rule_index += 1

    # Weighted Average
    sum_firing_strengths = np.sum(firing_strengths)

    if sum_firing_strengths == 0 or np.isnan(sum_firing_strengths):
        # Default output if no rules fire. Use the average of the output range [0, 1].
        predicted_power = np.mean(power_range_init) # Use the saved power range mean
    else:
        predicted_power = np.sum(firing_strengths * singleton_values) / sum_firing_strengths

    # Clamp output to the valid range [0, 1]
    predicted_power = max(0.0, min(1.0, predicted_power))

    return predicted_power

# Example of loading and using in a simulation loop:
# params_filename = 'trained_anfis_controller_info_numpy.json'
# try:
#     with open(params_filename, 'r') as f:
#         loaded_info = json.load(f)
# except FileNotFoundError:
#     print(f"Error loading trained controller: {params_filename} not found.")
#     # Handle this error in your simulation script (e.g., use PID instead)
#     exit() # Or sys.exit()


# trained_params = np.array(loaded_info['optimized_parameters'])
# loaded_num_mfs_error = loaded_info['num_mfs_error']
# loaded_num_mfs_delta_error = loaded_info['num_mfs_delta_error']
# loaded_mf_type = loaded_info['mf_type'] # 'trimf'
# loaded_and_method_str = loaded_info['and_method'] # 'min' or 'product'
# loaded_error_range_init = tuple(loaded_info['error_range_used_for_init'])
# loaded_delta_error_range_init = tuple(loaded_info['delta_error_range_used_for_init'])
# loaded_power_range_init = tuple(loaded_info['power_range'])

# # Get the correct MF and AND functions
# mf_func_loaded = triangular_mf # Or other MF functions if you added them
# and_method_func_loaded = get_and_method_func(loaded_and_method_str)


# # Inside your simulation loop (like the one in Step 3.1, but now with the ANFIS controller):
# # current_error_value = ... # Calculate error at current time step
# # current_delta_error_value = ... # Calculate delta error at current time step
#
# # Get control output from the trained ANFIS controller
# control_output = use_trained_sugeno(
#     current_error_value, current_delta_error_value,
#     trained_params, loaded_num_mfs_error, loaded_num_mfs_delta_error,
#     mf_func_loaded, and_method_func_loaded,
#     loaded_error_range_init, loaded_delta_error_range_init, loaded_power_range_init
# )
#
# # Use the control_output (heater power) in your system simulation for the next step.
# # ... simulate system step using control_output ...
"""