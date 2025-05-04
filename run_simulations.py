import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random
import json # Needed for loading ANFIS info
import sys # Needed for checking if ANFIS loading was successful

# --- Load Trained ANFIS Controller Function ---
# This function loads the parameters and structure from the JSON file
# and recreates the necessary components for the NumPy-based inference.
def load_trained_sugeno_controller(params_filename='trained_anfis_controller_info_numpy.json'):
    """
    Loads the trained Sugeno controller information from a JSON file
    and returns the necessary components for NumPy-based inference.
    Returns a dictionary containing the loaded info, or None if loading fails.
    """
    try:
        with open(params_filename, 'r') as f:
            loaded_info = json.load(f)
    except FileNotFoundError:
        print(f"Error loading controller: {params_filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error loading controller: Could not decode JSON from {params_filename}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading controller info: {e}")
        return None

    # Extract saved data
    try:
        trained_params = np.array(loaded_info['optimized_parameters'])
        num_mfs_error = loaded_info['num_mfs_error']
        num_mfs_delta_error = loaded_info['num_mfs_delta_error']
        mf_type = loaded_info['mf_type']
        and_method_str = loaded_info['and_method']
        error_range_init = tuple(loaded_info['error_range_used_for_init'])
        delta_error_range_init = tuple(loaded_info['delta_error_range_used_for_init'])
        power_range_init = tuple(loaded_info['power_range'])

        # Map the saved 'and_method' string back to the NumPy function
        if and_method_str == 'min':
            and_method_func = np.fmin
        elif and_method_str == 'product':
            and_method_func = np.multiply
        else:
            print(f"Error loading controller: Unsupported AND method string: {and_method_str}")
            return None

        # Assuming only triangular MFs were used and saved as 'trimf'
        if mf_type != 'trimf':
             print(f"Error loading controller: Unsupported MF type string: {mf_type}")
             return None
        # Need the triangular_mf function definition available for inference
        # We will define it again here for self-containment in this script.


    except KeyError as e:
        print(f"Error loading controller: Missing key in JSON file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while extracting data from JSON: {e}")
        return None

    print("Trained Sugeno controller information loaded successfully.")

    # Return a dictionary containing all necessary components for inference
    return {
        'trained_params': trained_params,
        'num_mfs_error': num_mfs_error,
        'num_mfs_delta_error': num_mfs_delta_error,
        'mf_type': mf_type,
        'and_method_func': and_method_func, # Pass the actual function
        'error_range_init': error_range_init,
        'delta_error_range_init': delta_error_range_init,
        'power_range_init': power_range_init,
        'mf_function': triangular_mf # Pass the MF function definition
    }

# --- Membership Function Implementation (Must match training) ---
# This function must be identical to the one used in train_anfis_controller.py
def triangular_mf(x, params):
    """Triangular membership function (a, b, c)."""
    x = np.asarray(x)
    a, b, c = params
    if a > b or b > c:
        return np.zeros_like(x, dtype=np.float64)
    if a == b and b == c:
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


# --- Trained Sugeno Inference Function (NumPy based) ---
# This function takes inputs and the loaded trained controller components
# and performs the fuzzy inference to get the output.
def use_trained_sugeno(input_error, input_delta_error, trained_controller_info):
    """
    Uses the trained Sugeno controller (NumPy implementation) to predict output for single inputs.
    Takes a dictionary of trained controller info as input.
    """
    # Extract necessary info from the loaded dictionary
    trained_params = trained_controller_info['trained_params']
    num_mfs_error = trained_controller_info['num_mfs_error']
    num_mfs_delta_error = trained_controller_info['num_mfs_delta_error']
    mf_function = trained_controller_info['mf_function']
    and_method_func = trained_controller_info['and_method_func']
    error_range_init = trained_controller_info['error_range_init']
    delta_error_range_init = trained_controller_info['delta_error_range_init']
    power_range_init = trained_controller_info['power_range_init']


    # Clamp inputs to the ranges used during training for consistency
    input_error_clamped = max(error_range_init[0], min(error_range_init[1], input_error))
    input_delta_error_clamped = max(delta_error_range_init[0], min(delta_error_range_init[1], input_delta_error))


    params = trained_params
    params_per_mf = 3 # Assuming 'trimf'

    num_input_mf_params = (num_mfs_error + num_mfs_delta_error) * params_per_mf
    num_singletons = num_mfs_error * num_mfs_delta_error

    # Unpack parameters (must match the order used during training/saving)
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
        predicted_power = np.mean(power_range_init)
    else:
        predicted_power = np.sum(firing_strengths * singleton_values) / sum_firing_strengths

    # Clamp output to the valid range [0, 1]
    predicted_power = max(0.0, min(1.0, predicted_power))

    return predicted_power


# --- System Parameters (Same as used for training data generation) ---
tau = 200       # Time constant in seconds.
K = 60          # System gain in °C / unit power (assuming P is 0 to 1).
Tamb_initial = 25 # Initial ambient temperature in °C.

# --- Simulation Parameters (Same as used for training data generation) ---
t_end = 8000     # Total simulation time in seconds.
dt_sim = 1       # Time step for simulation and data recording (s)
time_points = np.arange(0, t_end + dt_sim, dt_sim) # Time points for simulation steps

# --- Differential Equation (Our system model) ---
def temp_system(t, T, P, Tamb):
    """
    Describes the rate of change of temperature for the first-order system.
    dT/dt = (1/tau) * (K * P(t) + Tamb(t) - T(t))
    """
    dTdt = (1/tau) * (K * P + Tamb - T)
    return dTdt

# --- PID Controller Implementation (Same as used for training data generation) ---
# We define this as a function that takes current state and returns control output.

# PID gains (Use the SAME tuned gains from Step 3.1)
# These gains should be the ones that gave good performance in the PID simulation.
PID_Kp = 1.5
PID_Ki = 0.02
PID_Kd = 15.0

# PID state variables (need to be managed outside this function in the simulation loop)
# integral_error = 0
# last_error = 0
# last_measured_temp = Tamb_initial

def pid_controller(error, integral_error, derivative_measured_temp, Kp, Ki, Kd, dt):
    """Calculates the PID control output."""
    P_output = Kp * error + Ki * integral_error + Kd * derivative_measured_temp
    # Clamp the output to the physical limits [0, 1]
    P_output_clamped = max(0.0, min(1.0, P_output))
    return P_output_clamped

# --- Setpoint Profile (Same as used for training data generation) ---
setpoint_times = [0, 1000, 2500, 4000, 5500, 7000, t_end]
setpoint_values = [25, 50, 40, 65, 50, 55, 55]

def get_setpoint(t):
    """Gets the setpoint temperature at a given time t."""
    idx = np.searchsorted(setpoint_times, t, side='right') - 1
    return setpoint_values[idx]

# --- Ambient Temperature Disturbance (Same as used for training data generation) ---
Tamb_step_time_1 = t_end / 4
Tamb_step_amount_1 = 5
Tamb_step_time_2 = t_end * 3 / 4
Tamb_step_amount_2 = -3

def get_ambient_temp(t):
    """Gets the ambient temperature at a given time t, including step disturbances."""
    temp = Tamb_initial
    if t >= Tamb_step_time_1:
        temp += Tamb_step_amount_1
    if t >= Tamb_step_time_2:
        temp += Tamb_step_amount_2
    return temp

# --- Noise Parameter (Same as used for training data generation) ---
noise_std_dev = 0.5 # Standard deviation of Gaussian noise in °C.

# --- Unified Simulation Function ---

def run_simulation(controller_type, anfis_controller_info=None, pid_kp=None, pid_ki=None, pid_kd=None, output_filename='simulation_results.csv'):
    """
    Runs the temperature system simulation with the specified controller.

    Args:
        controller_type (str): 'ANFIS' or 'PID'.
        anfis_controller_info (dict, optional): Dictionary containing loaded ANFIS info if controller_type is 'ANFIS'. Defaults to None.
        pid_kp (float, optional): PID Proportional gain if controller_type is 'PID'. Defaults to None.
        pid_ki (float, optional): PID Integral gain if controller_type is 'PID'. Defaults to None.
        pid_kd (float, optional): PID Derivative gain if controller_type is 'PID'. Defaults to None.
        output_filename (str, optional): Name of the CSV file to save results. Defaults to 'simulation_results.csv'.
    """
    print(f"\n--- Running simulation with {controller_type} controller ---")

    # Initialize state variables for the system and controllers
    T_true_current = Tamb_initial
    # Initial measured temperature includes noise
    T_measured_current = T_true_current + np.random.normal(0, noise_std_dev)

    # State variables needed specifically for the PID controller and Delta Error calculation
    integral_error = 0
    last_error = 0
    last_measured_temp = Tamb_initial # For derivative calculation

    # Lists to store simulation data
    sim_data = {
        'Time': [],
        'Setpoint': [],
        'Measured_Temperature': [],
        'True_Temperature': [],
        'Ambient_Temp': [],
        'Controller_Output': [], # The power output from the controller (ANFIS or PID)
        'Error': [],
        'Delta_Error': []
    }

    # Simulation loop: Iterate through each time step
    for i in range(len(time_points) - 1):
        t = time_points[i] # Current time.
        t_next = time_points[i+1] # Time at the end of the current step.
        dt = dt_sim # Duration of the current time step.

        # Get current setpoint and ambient temp at the beginning of the step.
        setpoint = get_setpoint(t)
        ambient_temp = get_ambient_temp(t)

        # Calculate current error (based on measured temperature, as the controller sees)
        error = setpoint - T_measured_current

        # Calculate Delta Error (change in error over the last time step)
        delta_error = error - last_error

        # Calculate derivative of measured temperature (used by PID D-term and potentially ANFIS input)
        if i > 0:
             derivative_measured_temp = (T_measured_current - last_measured_temp) / dt
        else:
             derivative_measured_temp = 0 # Assume no initial rate of change at the first step.

        # --- Get Control Output from the specified controller ---
        controller_output = 0.0 # Initialize controller output

        if controller_type == 'ANFIS':
            if anfis_controller_info is None:
                print("Error: ANFIS controller info not provided.")
                return

            # ANFIS controller takes Error and Delta_Error as inputs
            # Use the trained Sugeno inference function with the current inputs and loaded info.
            controller_output = use_trained_sugeno(
                error,
                delta_error, # Use delta_error as input, matching training data structure
                anfis_controller_info
            )


        elif controller_type == 'PID':
            if pid_kp is None or pid_ki is None or pid_kd is None:
                print("Error: PID gains not provided.")
                return

            # PID controller uses error, integral_error, and derivative_measured_temp
            # Update integral error for the PID controller's state
            integral_error += error * dt

            # Calculate PID output using the PID function
            controller_output = pid_controller(
                error,
                integral_error,
                derivative_measured_temp,
                pid_kp, pid_ki, pid_kd,
                dt
            )

        else:
            print(f"Error: Unknown controller type '{controller_type}'.")
            return

        # --- Simulate System Step ---
        # Use the calculated controller output (power) to simulate the system's response for one time step.
        sol_step = solve_ivp(
            lambda t, T: temp_system(t, T[0], controller_output, ambient_temp),
            [t, t_next],       # Solve from current time 't' to next time 't_next'.
            [T_true_current],  # Start the ODE solution from the current true temperature.
            dense_output=True, # Allow evaluation at intermediate points if needed (not strictly used here).
            t_eval=[t_next]    # Get the solution specifically at the end of the time step 't_next'.
        )
        T_true_next = sol_step.y[0][0] # Extract the true temperature at the end of the step.

        # Add noise to the true temperature at the end of the step to get the measured temperature for the NEXT step.
        noise = np.random.normal(0, noise_std_dev)
        T_measured_next = T_true_next + noise

        # --- Store data for this time step ---
        sim_data['Time'].append(t)
        sim_data['Setpoint'].append(setpoint)
        sim_data['Measured_Temperature'].append(T_measured_current)
        sim_data['True_Temperature'].append(T_true_current)
        sim_data['Ambient_Temp'].append(ambient_temp)
        sim_data['Controller_Output'].append(controller_output)
        sim_data['Error'].append(error)
        sim_data['Delta_Error'].append(delta_error)


        # --- Update state variables for next iteration ---
        last_error = error # Store current error to calculate delta_error in the next step.
        last_measured_temp = T_measured_current # Store current measured temp for derivative calculation in the next step.
        T_true_current = T_true_next # Update the true temperature for the next step.
        T_measured_current = T_measured_next # Update the measured temperature for the next step.


    # --- Save Simulation Data ---
    df_sim_results = pd.DataFrame(sim_data)
    try:
        df_sim_results.to_csv(output_filename, index=False)
        print(f"Simulation results saved to {output_filename}")
    except IOError as e:
        print(f"Error saving simulation results to {output_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving simulation results: {e}")


# --- Main execution for running simulations ---

# 1. Run simulation with Trained ANFIS Controller
# Load the trained controller information from the JSON file.
anfis_controller_info = load_trained_sugeno_controller('trained_anfis_controller_info_numpy.json')

if anfis_controller_info: # Check if ANFIS controller info was loaded successfully
    run_simulation(
        controller_type='ANFIS',
        anfis_controller_info=anfis_controller_info, # Pass the loaded info dictionary
        output_filename='anfis_simulation_results.csv'
    )
else:
    print("Skipping ANFIS simulation due to loading error.")
    # Optionally exit or handle this case as needed if ANFIS is critical.
    # sys.exit("ANFIS controller loading failed.")


# 2. Run simulation with PID Controller (using the same tuned gains)
# The run_simulation function handles initial conditions and PID state variables internally,
# so we just need to call it with the PID parameters.
run_simulation(
    controller_type='PID',
    pid_kp=PID_Kp,
    pid_ki=PID_Ki,
    pid_kd=PID_Kd,
    output_filename='pid_simulation_results.csv'
)

print("\nSimulations complete. You can now run the comparison script.")
