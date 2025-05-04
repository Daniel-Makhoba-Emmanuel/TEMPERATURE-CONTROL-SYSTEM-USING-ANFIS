import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random

# --- System Parameters (Based on our previous agreement) ---
tau = 200       # Time constant in seconds.
K = 60          # System gain in °C / unit power (assuming P is 0 to 1).
Tamb_initial = 25 # Initial ambient temperature in °C.

# --- Simulation Parameters ---
t_end = 8000     # Total simulation time in seconds.
dt_sim = 1       # Time step for simulation and data recording (s)
time_points = np.arange(0, t_end + dt_sim, dt_sim) # Time points for simulation steps

# --- Differential Equation (Our system model) ---
def temp_system(t, T, P, Tamb):
    """
    Describes the rate of change of temperature for the first-order system.
    dT/dt = (1/tau) * (K * P + Tamb - T)
    """
    dTdt = (1/tau) * (K * P + Tamb - T)
    return dTdt

# --- PID Controller Parameters (Needs Tuning!) ---
# These are initial guesses. YOU MUST TUNE THESE by observing the simulation plots.
# Aim for a stable, reasonably responsive control without excessive oscillations or overshoot.
Kp = 1.5  # Proportional Gain - Impacts response speed and steady-state error (indirectly)
Ki = 0.02 # Integral Gain - Helps eliminate steady-state error
Kd = 15.0 # Derivative Gain - Helps damp oscillations and improve response time to changes

# --- PID Controller State Variables ---
integral_error = 0
last_error = 0
# Initialize last_temp for derivative calculation.
# We'll use the measured temperature for the derivative, which is more realistic.
last_measured_temp = Tamb_initial

# --- Define Setpoint Profile ---
# Setpoint changes ensure the PID (and thus ANFIS data) covers heating up, cooling down, and steady states.
setpoint_times = [0, 1000, 2500, 4000, 5500, 7000, t_end] # Times when setpoint changes
setpoint_values = [25, 50, 40, 65, 50, 55, 55]        # Corresponding setpoint temperatures (°C)

def get_setpoint(t):
    """Gets the setpoint temperature at a given time t."""
    idx = np.searchsorted(setpoint_times, t, side='right') - 1
    return setpoint_values[idx]

# --- Define Ambient Temperature Disturbance (Same as before) ---
Tamb_step_time_1 = t_end / 4      # Time of the first step change
Tamb_step_amount_1 = 5          # Amount of first step change in °C (e.g., 25 -> 30)
Tamb_step_time_2 = t_end * 3 / 4  # Time of the second step change
Tamb_step_amount_2 = -3         # Amount of second step change in °C (e.g., 30 -> 27)

def get_ambient_temp(t):
    """Gets the ambient temperature at a given time t, including step disturbances."""
    temp = Tamb_initial
    if t >= Tamb_step_time_1:
        temp += Tamb_step_amount_1
    if t >= Tamb_step_time_2:
        temp += Tamb_step_amount_2
    return temp

# --- Noise Parameter (Same as before) ---
noise_std_dev = 0.5 # Standard deviation of Gaussian noise in °C.

# --- Simulation with PID Controller (Step-by-Step) ---

# Lists to store data for ANFIS training CSV
anfis_data = {
    'Time': [],
    'Error': [],
    'Delta_Error': [],
    'PID_Output_Power': [], # This will be the target output for ANFIS
    'Measured_Temperature': [], # The noisy temperature the controller sees
    'True_Temperature': [], # The actual system temperature (without noise)
    'Setpoint': [],
    'Ambient_Temp': []
}

# Initial conditions
T_true_current = Tamb_initial # Start true temperature at initial ambient
# Simulate initial measurement noise
T_measured_current = T_true_current + np.random.normal(0, noise_std_dev)


print("Simulating system under PID control to generate ANFIS training data...")

# Loop through each time step
for i in range(len(time_points) - 1):
    t = time_points[i]
    t_next = time_points[i+1]
    dt = dt_sim # Time step duration

    # Get current setpoint and ambient temp
    setpoint = get_setpoint(t)
    ambient_temp = get_ambient_temp(t)

    # --- PID Controller Calculation ---
    # Controller uses the MEASURED temperature
    error = setpoint - T_measured_current
    anfis_data['Error'].append(error)
    anfis_data['Setpoint'].append(setpoint)
    anfis_data['Ambient_Temp'].append(ambient_temp)
    anfis_data['Measured_Temperature'].append(T_measured_current)
    anfis_data['True_Temperature'].append(T_true_current) # Store true temp for plotting/analysis
    anfis_data['Time'].append(t)


    # Calculate Delta Error (using backward difference on error)
    # Or calculate derivative of measured temperature (more robust for derivative term)
    # Let's calculate the derivative of the measured temperature for the Kd term,
    # and delta_error = error - last_error for the Delta_Error input to ANFIS.

    delta_error = error - last_error # Change in error over the last time step

    if i > 0:
         # Derivative of measured temperature
         derivative_measured_temp = (T_measured_current - last_measured_temp) / dt
         # The derivative of error is -derivative of temperature if setpoint is constant
         # When setpoint changes, de/dt is tricky. Using -dT/dt is often better for the Kd term.
         derivative_term_input = -derivative_measured_temp
    else:
         derivative_term_input = 0 # Assume no initial rate of change


    anfis_data['Delta_Error'].append(delta_error) # Store delta_error for ANFIS input


    # Integral error
    integral_error += error * dt

    # PID Output (Control Signal P)
    # Use error for P and I terms, derivative of measured temperature for D term
    P_output = Kp * error + Ki * integral_error + Kd * derivative_term_input

    # Clamp the output to the physical limits of the heater (0 to 1)
    P_output_clamped = max(0.0, min(1.0, P_output))
    anfis_data['PID_Output_Power'].append(P_output_clamped)


    # --- Simulate System Step (using the clamped PID output) ---
    # We need to solve the ODE for one time step dt using the calculated power
    sol_step = solve_ivp(
        lambda t, T: temp_system(t, T[0], P_output_clamped, ambient_temp),
        [t, t_next],
        [T_true_current], # Start ODE from the true current temperature
        dense_output=True,
        t_eval=[t_next] # Get solution at the end of the step
    )
    T_true_next = sol_step.y[0][0] # True temperature at the end of the step

    # Add noise to the measured temperature for the next step's measurement
    noise = np.random.normal(0, noise_std_dev)
    T_measured_next = T_true_next + noise

    # Update state variables for next iteration
    last_error = error
    last_measured_temp = T_measured_current # Use the measured temperature from this step
    T_true_current = T_true_next # Update true temperature for the next step
    T_measured_current = T_measured_next # Update measured temperature for the next step


print("PID simulation complete.")

# --- Create DataFrame and Save Data ---
df_anfis_training = pd.DataFrame(anfis_data)

csv_filename_anfis = 'anfis_training_data_pid.csv'
df_anfis_training.to_csv(csv_filename_anfis, index=False)
print(f"ANFIS training data saved to {csv_filename_anfis}")

# --- Plot Results of PID Simulation (for tuning) ---
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(df_anfis_training['Time'], df_anfis_training['Measured_Temperature'], label='Measured Temperature (with noise)')
plt.plot(df_anfis_training['Time'], df_anfis_training['True_Temperature'], label='True Temperature (no noise)', linestyle=':', alpha=0.6)
plt.plot(df_anfis_training['Time'], df_anfis_training['Setpoint'], label='Setpoint', linestyle='--')
plt.plot(df_anfis_training['Time'], df_anfis_training['Ambient_Temp'], label='Ambient Temperature', linestyle=':', color='gray')
plt.ylabel('Temperature (°C)')
plt.title('System under PID Control (Data Generation for ANFIS)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df_anfis_training['Time'], df_anfis_training['PID_Output_Power'] * 100, label='PID Output Power (%)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Power (%)')
plt.title('PID Controller Output')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df_anfis_training['Time'], df_anfis_training['Error'], label='Temperature Error (°C)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Error (°C)')
plt.title('Temperature Error')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()