import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random # Needed for random intervals

# --- System Parameters (Matching agreed-upon values) ---
tau = 200       # Time constant in seconds.
K = 60          # System gain in °C / unit power (assuming P is 0 to 1).
Tamb_initial = 25 # Initial ambient temperature in °C.

# --- Simulation Parameters ---
t_end = 8000     # Total simulation time in seconds. Increased to accommodate dynamics and disturbances.
dt_sim = 1       # Time step for simulation output data (s)
t_eval = np.arange(0, t_end, dt_sim) # Time points where we want the solution

# --- Differential Equation ---
def temp_system(t, T, P_func, Tamb_func):
    """
    Describes the rate of change of temperature for the first-order system.
    dT/dt = (1/tau) * (K * P(t) + Tamb(t) - T(t))
    """
    P = P_func(t)
    Tamb = Tamb_func(t)
    dTdt = (1/tau) * (K * P + Tamb - T)
    return dTdt

# --- Define Input Signals and Disturbances over time ---

# 1. Input Power P(t): Random Step Changes with Random Durations
min_power_interval = tau / 2 # Minimum duration for a power level
max_power_interval = 2 * tau # Maximum duration for a power level

power_times = [0] # Start at time 0
power_levels = []
current_time = 0

# Generate random power levels and times until we reach t_end
while current_time < t_end:
    # Choose a random power level between 0 and 1
    power_levels.append(random.random())
    # Choose a random duration for this power level
    duration = random.uniform(min_power_interval, max_power_interval)
    current_time += duration
    # Add the end time of this interval (or t_end if it exceeds it)
    power_times.append(min(current_time, t_end))

# Ensure the last power level applies until t_end
if power_times[-1] < t_end:
     power_times.append(t_end)
     power_levels.append(power_levels[-1] if power_levels else random.random()) # Add one more if needed

# power_times now contains the transition points [0, t1, t2, ..., t_end]
# power_levels contains the level active from power_times[i] to power_times[i+1]


def P_func(t):
    """
    Defines the power input signal P(t) using random step changes with random durations.
    """
    # Find which interval the current time t falls into
    # searchsorted finds the index where t would be inserted to maintain order
    # Subtract 1 to get the index of the interval start time
    idx = np.searchsorted(power_times, t, side='right') - 1
    # Ensure index is within the valid range for power_levels
    idx = max(0, min(idx, len(power_levels) - 1))
    return power_levels[idx]


# 2. Ambient Temperature Tamb(t): Two Step Changes
Tamb_step_time_1 = t_end / 4      # Time of the first step change
Tamb_step_amount_1 = 5          # Amount of first step change in °C
Tamb_step_time_2 = t_end * 3 / 4  # Time of the second step change
Tamb_step_amount_2 = -3         # Amount of second step change in °C (e.g., a drop)


def Tamb_func(t):
    """
    Defines the ambient temperature Tamb(t) with two step changes.
    """
    temp = Tamb_initial
    if t >= Tamb_step_time_1:
        temp += Tamb_step_amount_1
    if t >= Tamb_step_time_2:
        temp += Tamb_step_amount_2
    return temp

# --- Initial Condition ---
T_initial = Tamb_initial # Start at ambient temperature

# --- Solve the Ordinary Differential Equation (Simulate System) ---
print(f"Simulating system for {t_end} seconds...")

sol = solve_ivp(
    lambda t, T: temp_system(t, T, P_func, Tamb_func), # Function defining the ODE
    [0, t_end],      # Time span for the simulation
    [T_initial],     # Initial condition [T(0)]
    dense_output=True, # Allows evaluating solution at arbitrary points
    t_eval=t_eval    # Specific time points where we want the solution
)

print("Simulation complete.")

# Extract results
time_sim = sol.t
temp_true = sol.y[0] # The 'true' temperature without noise

# --- Add Simulated Sensor Noise ---
noise_mean = 0       # Mean of the noise
noise_std_dev = 0.5 # Standard deviation of the noise in °C.
noise = np.random.normal(noise_mean, noise_std_dev, len(temp_true))
temp_noisy = temp_true + noise

# --- Get the values of P(t) and Tamb(t) at the simulation time points ---
# Need to call the functions again for the specific time points from the solver
power_sim = np.array([P_func(t) for t in time_sim])
tamb_sim = np.array([Tamb_func(t) for t in time_sim])

# --- Store Data ---
data = pd.DataFrame({
    'Time': time_sim,
    'Input_Power': power_sim,
    'Ambient_Temp': tamb_sim,
    'Temperature': temp_noisy # Use the noisy temperature as our sensor reading
})

# --- Save Data to CSV ---
csv_filename = 'temperature_training_data.csv'
data.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# --- Plot Data to Visualize ---
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1) # Top subplot for Temperature
plt.plot(data['Time'], data['Temperature'], label='Simulated Temperature (with noise)')
plt.plot(data['Time'], temp_true, label='True Temperature (no noise)', linestyle='--', alpha=0.7)
plt.plot(data['Time'], data['Ambient_Temp'], label='Ambient Temperature', linestyle=':', color='gray')
plt.ylabel('Temperature (°C)')
plt.title('Simulated Temperature System Data')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2) # Middle subplot for Input Power
# Scale power for clarity if P is 0-1, assuming max power is 100%
plt.plot(data['Time'], data['Input_Power'] * 100, label='Input Power (%)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Input Power (%)')
plt.title('Input Power Applied to System')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3) # Bottom subplot for Ambient Temperature
plt.plot(data['Time'], data['Ambient_Temp'], label='Ambient Temperature', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Ambient Temperature Disturbance')
plt.legend()
plt.grid(True)


plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()