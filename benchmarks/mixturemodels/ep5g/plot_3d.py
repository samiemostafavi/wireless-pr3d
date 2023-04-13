import json
import os
import warnings
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from pr3d.de import ConditionalGaussianMixtureEVM, ConditionalGaussianMM
from matplotlib import ticker, cm, colors

warnings.filterwarnings("ignore")

import scienceplots
plt.style.use(['science','ieee'])

exp_args = {
    "label" : "ep5g/3dplot",
    "target_y_points" : [5,50,10],
    "target_delay" : 7,
    "loc_y_points" : [0,3.1,0.1],
    "loc_x_points" : [0,4.1,0.1],
    "target" : "send",
    "target_mean" : 5433109.863742597,
    "target_scale" : 1e-6,
    "model" : ["ep5g/trained_loc_send_comp","gmevm","0"],
    
}

sampling_points_x = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
sampling_points_y = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]

if __name__ == "__main__":


    # this project folder setting
    p = Path(__file__).parents[1]
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    key_label = exp_args["target"]
    key_mean = exp_args["target_mean"]
    key_scale = exp_args["target_scale"]
    
    
    #ax = fig.add_subplot(111, projection='3d')

    # plot predictions
    model_list = exp_args["model"]
    model_project_name = model_list[0]
    model_conf_key = model_list[1]
    ensemble_num = model_list[2]
    model_path = (
        main_path + model_project_name + "_results/" + model_conf_key + "/"
    )

    with open(
        model_path + f"model_{ensemble_num}.json"
    ) as json_file:
        model_dict = json.load(json_file)

    if model_dict["type"] == "gmm":
        pr_model = ConditionalGaussianMM(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
    elif model_dict["type"] == "gmevm":
        pr_model = ConditionalGaussianMixtureEVM(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )

    X = np.arange(exp_args["loc_x_points"][0], exp_args["loc_x_points"][1], exp_args["loc_x_points"][2])
    Y = np.arange(exp_args["loc_y_points"][0], exp_args["loc_y_points"][1], exp_args["loc_y_points"][2])
    x_arr = X
    y_arr = Y
    X, Y = np.meshgrid(X, Y)

    # define x numpy list
    x_list = []
    for x in x_arr:
        for y in y_arr:
            x_list.append(
                [ x, y ]
            )
    x = np.array(x_list)

    # define y points and run the inference
    y = np.repeat(exp_args["target_delay"]-(key_mean*key_scale), len(x))
    y = np.array(y,dtype=np.float64)

    #y = y.clip(min=0.00)
    prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
    pred_tail = np.float64(1.00)-np.array(pred_cdf,dtype=np.float64)
    pred_tail = np.reshape(pred_tail,(len(x_arr),len(y_arr)))

    # Alternatively, you can manually set the levels
    # and the norm:
    #lev_exp = np.arange(np.floor(np.log10(pred_cdf.min())-1),
    #                    np.ceil(np.log10(pred_cdf.max())+1),
    #                    0.2,
    #                )
    #levs = np.power(10, lev_exp)
    fig, ax = plt.subplots()
    #cs = ax.contourf(X, Y, pred_cdf, levs, norm=colors.LogNorm())

    #surf = ax.plot_surface(X, Y, pred_cdf, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)

    # create the figure
    #fig = plt.figure()
    #ax.contour(X, Y, pred_cdf, cmap='RdGy')
    cs = plt.contourf(X, Y, np.transpose(pred_tail), locator=ticker.LogLocator(), cmap=cm.coolwarm)
    
    #ax.set_zlim3d(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    

    #fig.colorbar(surf, shrink=0.5, aspect=10)

            
    ax.set_ylim(exp_args["loc_y_points"][0],exp_args["loc_y_points"][1]-exp_args["loc_y_points"][2])
    ax.set_xlim(exp_args["loc_x_points"][0],exp_args["loc_x_points"][1]-exp_args["loc_x_points"][2])

    ax.set_yticks(np.arange(exp_args["loc_y_points"][0],exp_args["loc_y_points"][1],1), labels=None)
    ax.set_xticks(np.arange(exp_args["loc_x_points"][0],exp_args["loc_x_points"][1],1), labels=None)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax.set_aspect('equal')

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(cs,fraction=0.0345, pad=0.05)


    ax.scatter(sampling_points_x,sampling_points_y)

    # cdf figure
    #fig.tight_layout()
    fig.savefig(project_path + f"{exp_args['target_delay']}ms_{key_label}.png")

   