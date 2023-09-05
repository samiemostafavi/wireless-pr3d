import matplotlib.pyplot as plt
import numpy as np
import json 
from pathlib import Path

# List of JSON file paths
json_file_paths = [
    "timeseries/newr1ep5g_results/s1s2s3/evaluations/result_230825151134_876091.json",
    "timeseries/newr1ep5g_results/s1s2s3/evaluations/result_230831161655_456352.json",
]
res_file_path = "timeseries/newr1ep5g_results/s1s2s3/evaluations/"

# Initialize lists to store baseline_values, result_values, and errors
baseline_values_list = []
result_values_list = []
errors_list = []
ids = []

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
        ids.append(data["id"])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot errors for each file
for i in range(len(json_file_paths)):
    plt.plot(baseline_values_list[i], errors_list[i], marker="o", label=f"ID: {ids[i]}")

plt.xscale("log")  # Set x-axis to logarithmic scale
plt.xlabel("Baseline (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error in Log Scale")

# Reverse the x-axis while keeping it in logarithmic scale
plt.gca().invert_xaxis()

plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.savefig(Path(res_file_path) / "res.png")
