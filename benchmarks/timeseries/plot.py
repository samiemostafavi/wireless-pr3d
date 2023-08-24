import matplotlib.pyplot as plt
import numpy as np
import json 
from pathlib import Path

# List of JSON file paths
json_file_paths = [
    "timeseries/r1ep5g_results/evaluations/result_230823180213_286720.json",
    "timeseries/r1ep5g_results/evaluations/result_230823141843_796856.json",
    "timeseries/r1ep5g_results/evaluations/result_230823102952_699741.json"
]
res_file_path = "timeseries/r1ep5g_results/evaluations/"

# Initialize lists to store baseline_values, result_values, and errors
baseline_values_list = []
result_values_list = []
errors_list = []

# Read each JSON file and extract relevant data
for json_path in json_file_paths:
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        result_dict = data["result"]
        
        # Extract baseline_values and result_values from the current file
        baseline_values = 1.00 - np.array(list(map(float, result_dict.keys())))
        result_values = np.array(list(result_dict.values()))

        # Calculate errors in logarithmic scale
        errors = np.log10(result_values) - np.log10(baseline_values)

        baseline_values_list.append(baseline_values)
        result_values_list.append(result_values)
        errors_list.append(errors)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot errors for each file
for i in range(len(json_file_paths)):
    plt.plot(baseline_values_list[i], errors_list[i], marker="o", label=f"File {i+1}")

plt.xscale("log")  # Set x-axis to logarithmic scale
plt.xlabel("Baseline (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error in Log Scale")

# Reverse the x-axis while keeping it in logarithmic scale
plt.gca().invert_xaxis()

plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig(Path(res_file_path) / "res.png")
