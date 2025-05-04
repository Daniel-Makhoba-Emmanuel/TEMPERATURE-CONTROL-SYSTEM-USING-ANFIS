import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys # To exit if data loading fails

# --- File names for simulation results ---
anfis_results_file = 'anfis_simulation_results.csv'
pid_results_file = 'pid_simulation_results.csv'

# --- Load simulation results ---
try:
    df_anfis = pd.read_csv(anfis_results_file)
    print(f"Successfully loaded ANFIS results from {anfis_results_file}")
except FileNotFoundError:
    print(f"Error: {anfis_results_file} not found. Run the simulation script ('run_simulations.py') first.")
    sys.exit(f"Required simulation results file not found: {anfis_results_file}") # Exit if file not found
except Exception as e:
    print(f"An error occurred while loading ANFIS data: {e}")
    sys.exit(f"Failed to load data from {anfis_results_file}")


try:
    df_pid = pd.read_csv(pid_results_file)
    print(f"Successfully loaded PID results from {pid_results_file}")
except FileNotFoundError:
    print(f"Error: {pid_results_file} not found. Run the simulation script ('run_simulations.py') first.")
    sys.exit(f"Required simulation results file not found: {pid_results_file}") # Exit if file not found
except Exception as e:
    print(f"An error occurred while loading PID data: {e}")
    sys.exit(f"Failed to load data from {pid_results_file}")


# --- Comparison Plots ---

# Plot 1: Temperature Response Comparison
plt.figure(figsize=(14, 8)) # Set figure size for better readability
plt.plot(df_anfis['Time'], df_anfis['Measured_Temperature'], label='ANFIS Measured Temp', color='blue', linewidth=1.5)
plt.plot(df_pid['Time'], df_pid['Measured_Temperature'], label='PID Measured Temp', color='red', linestyle='--', linewidth=1.5)
plt.plot(df_anfis['Time'], df_anfis['Setpoint'], label='Setpoint', color='green', linestyle='-.', linewidth=1.5, alpha=0.7)
plt.plot(df_anfis['Time'], df_anfis['Ambient_Temp'], label='Ambient Temp', color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Response Comparison: ANFIS vs. PID Control')
plt.legend()
plt.grid(True)

# Plot 2: Controller Output Comparison (Heater Power)
plt.figure(figsize=(14, 6))
plt.plot(df_anfis['Time'], df_anfis['Controller_Output'] * 100, label='ANFIS Output Power (%)', color='blue', linewidth=1.5)
plt.plot(df_pid['Time'], df_pid['Controller_Output'] * 100, label='PID Output Power (%)', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Power (%)')
plt.title('Controller Output Comparison: ANFIS vs. PID Control')
plt.legend()
plt.grid(True)

# Plot 3: Temperature Error Comparison
plt.figure(figsize=(14, 6))
plt.plot(df_anfis['Time'], df_anfis['Error'], label='ANFIS Error (°C)', color='blue', linewidth=1.5)
plt.plot(df_pid['Time'], df_pid['Error'], label='PID Error (°C)', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Error (°C)')
plt.title('Temperature Error Comparison: ANFIS vs. PID Control')
plt.legend()
plt.grid(True)


plt.tight_layout() # Automatically adjust subplot parameters for a tight layout.
plt.show() # Display all generated plots.

# --- Performance Metrics ---

# Calculate RMSE (Root Mean Squared Error) of the temperature error for each controller
# RMSE = sqrt(mean(Error^2))
# This metric quantifies the average magnitude of the temperature error over the simulation.
rmse_anfis = np.sqrt(np.mean(df_anfis['Error']**2))
rmse_pid = np.sqrt(np.mean(df_pid['Error']**2))

print("\n--- Performance Metrics (Lower RMSE indicates better performance) ---")
print(f"ANFIS Controller RMSE (Temperature Error): {rmse_anfis:.4f} °C")
print(f"PID Controller RMSE (Temperature Error):   {rmse_pid:.4f} °C")

# You can add other metrics here for a more detailed comparison, such as:
# - Mean Absolute Error (MAE): Average of the absolute errors. Less sensitive to outliers than RMSE.
#   mae_anfis = np.mean(np.abs(df_anfis['Error']))
#   mae_pid = np.mean(np.abs(df_pid['Error']))
#   print(f"ANFIS Controller MAE (Temperature Error): {mae_anfis:.4f} °C")
#   print(f"PID Controller MAE (Temperature Error):   {mae_pid:.4f} °C")
#
# - Integrated Squared Error (ISE): Sum of squared errors over time. Penalizes large errors heavily.
#   ise_anfis = np.sum(df_anfis['Error']**2) * (df_anfis['Time'].iloc[1] - df_anfis['Time'].iloc[0]) # Multiply by time step
#   ise_pid = np.sum(df_pid['Error']**2) * (df_pid['Time'].iloc[1] - df_pid['Time'].iloc[0])
#   print(f"ANFIS Controller ISE: {ise_anfis:.2f}")
#   print(f"PID Controller ISE:   {ise_pid:.2f}")
#
# - Metrics like Rise Time, Settling Time, and Overshoot require more complex analysis of the temperature response curve,
#   often by identifying setpoint change points and analyzing the response segments.

print("\nComparison complete. Analyze the plots and metrics to evaluate the performance of the trained ANFIS controller relative to the PID controller.")
