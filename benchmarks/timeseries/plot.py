import matplotlib.pyplot as plt
import numpy as np
import json 
from pathlib import Path

# List of JSON file paths
json_file_paths = [
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912111633_481481.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912134732_962676.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912134831_267665.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912134916_249222.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912135000_125001.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912135052_125124.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912113517_170170.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912135212_134752.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912141134_368337.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912143001_901032.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912144844_304827.json",
    "timeseries/newr1ep5g_results/s1/evaluations/result_230912150924_586978.json"
]
res_file_path = "timeseries/newr1ep5g_results/s1/evaluations/"

plotconf = {
    '230912083511_248506': {
        'label': 'non-conditional recurrent',
        'color':'C0',
    },
    '230912082335_799014': {
        'label': 'non-conditional nonrecurrent',
        'color':'C2'
    }
}


# Initialize lists to store baseline_values, result_values, and errors
results_dict = {}


#baseline_values_list = []
#result_values_list = []
#errors_list = []
#modelids = set()
#parquet_files = set()

# Read each JSON file and extract relevant data
for json_path in json_file_paths:
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        model_id = data["model"]["id"]
        parquet_file= data["data"]["parquet_files"]
        result_dict = data["result"]
        #modelids.add(model_id)
        #parquet_files.add(parquet_file)
        
        # Extract baseline_values and result_values from the current file
        baseline_values = 1.00 - np.array(list(map(float, result_dict.keys())))
        result_values = 1.00 - np.array(list(result_dict.values()))

        # Calculate errors in logarithmic scale
        errors = np.log10(result_values) - np.log10(baseline_values)

        #baseline_values_list.append(baseline_values)
        #result_values_list.append(result_values)
        #errors_list.append(errors)
        if model_id in results_dict.keys():
            results_dict[model_id]["error_values"].append(list(errors))
        else:
            results_dict[model_id] = {"error_values":[]}


#print(results_dict)

modelids = list(results_dict.keys())
# Initialize lists to store data for box plots
box_data = {i: [] for i in range(len(baseline_values))}

# Extract data for box plots
for i in range(len(baseline_values)):
    for key in modelids:
        box_data[i].append([arr[i] for arr in results_dict[key]['error_values']])

# Create box plot
fig, ax = plt.subplots()

barwidth = 0.2
bardis = 0.05

num_bars = len(modelids)
sectionwidth = (barwidth+bardis)*float(num_bars) - bardis
sectionstart = -sectionwidth*0.5
x_pos = []
for bar in range(num_bars):
    x_pos.append(sectionstart+(barwidth*0.5)+bar*(barwidth+bardis))

bps = []
labels = []
for i, modelid in enumerate(modelids):
    for j in range(len(baseline_values)):
        bp = ax.boxplot(
            abs(np.array(box_data[j][i])), 
            positions=[j+1 + x_pos[i]], 
            widths=barwidth,
            patch_artist=True,
            boxprops=dict(facecolor=plotconf[modelid]['color'])
        )
    bps.append(bp["boxes"][0])
    labels.append(plotconf[modelid]['label'])

ax.legend(bps, labels)

# Set labels and title

#xticks = [i + (barwidth/2.0) for i in range(1, len(baseline_values)+1)]
#print(xticks)
ax.set_xticks(np.array(range(len(baseline_values)))+1)
ax.set_xticklabels(1.00 - np.array(baseline_values))

plt.xlabel("Quantile")
plt.ylabel("Error (log scale)")
plt.grid(True)
plt.tight_layout()

"""
# Set labels and title
xticks = [i + 0.05 for i in range(1, 5)]
ax.set_xticks(xticks)
ax.set_xticklabels(['1', '2', '3'])

modelids = list(results_dict.keys())
for model_id in modelids:
    for idx, baseline_value in enumerate(baseline_values):
        results_dict[model_id][str(idx)] = list(map(lambda x: x[idx], zip(*results_dict[model_id]["error_values"])))

# Create the plot
plt.figure(figsize=(10, 6))

# Set x-positions for boxes
x_pos_range = np.arange(len(modelids)) / (len(modelids) - 1)
x_pos = (x_pos_range * 0.5) + 0.75

# Plot errors for each file
for idx, model_id in enumerate(modelids):
    data = results_dict[model_id]["error_values"]
    bp = plt.boxplot(
        np.array(data), sym='', whis=[0, 100], widths=0.6 / len(modelids),
        positions=[x_pos[idx] + j * 1 for j in range(len(data.T))]
    )

    plt.boxplot(results_dict[model_id]["error_values"])
    #plt.plot(baseline_values, abs(np.array(results_dict[modelids[i]]["avg_errors"])), marker="o", label=f"ID: {modelids[i]}")

plt.xticks([1, 2, 3], baseline_values)

plt.xscale("log")  # Set x-axis to logarithmic scale
plt.xlabel("Baseline (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error in Log Scale")

# Reverse the x-axis while keeping it in logarithmic scale
plt.gca().invert_xaxis()

plt.grid(True)
plt.legend()
plt.tight_layout()
"""
# Show the plot
plt.savefig(Path(res_file_path) / "res.png")
