[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Wireless latency prediction code


## Start

Create a Python 3.9 virtual environment using `virtualenv`

```
cd benchmarks
python -m virtualenv --python=python3.9 ./venv
source venv/bin/activate
```

Install dependencies
```
sudo apt-get install openjdk-8-jdk
pip install numpy==1.23.3
pip install pandas==1.5.0
pip install gdown
pip install -Ur requirements.txt
```

To use SciencePlot, latex must be installed on your machine
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

### Ericsson Private 5G Results

First figure of the paper, downlink latency measurements
```
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-pdf -r 0 -c 0 -y 3,300,30
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-tail --log -r 0 -c 0 -y 3,300,30
```

Download the parquet files, store them in `mixturemodels/ep5g/measurement_results`.
```
cd ~/wireless-latency-prediction-project/benchmarks/mixturemodels/ep5g
gdown https://drive.google.com/uc?id=1bBJjjCZcPmdiKYPGmiTkDJGfsZiVVXhn
unzip ep5g_measurements.zip
cd ~/wireless-latency-prediction-project/benchmarks
```

Open all the `parquet` files and show a preview of them:
```
python -m mixturemodels prep_dataset -d ep5g/measurement_training --preview
python -m mixturemodels prep_dataset -d ep5g/measurement_validation --preview
```

Important: copy all the training parquet files to the validation folder. Because when we prepare validation dataset, it needs to have a full range of samples for a consistent standardization.
Prepare the dataset by the conditions defined in json format with `-x` argument. You can normalize features using `-n`.
Use `-s` to set a size for all conditional datasets. In order to even out the number of conditional samples.
```
python -m mixturemodels prep_dataset -d ep5g/measurement_training -x '{"X":[0.0,1.0,2.0,3.0,4.0],"Y":[0.0,1.0,2.0,3.0]}' -l ep5g/prepped_loc -n X,Y
python -m mixturemodels prep_dataset -d ep5g/measurement_validation -x '{"X":[0.0,0.5,1.0,2.0,3.0,4.0],"Y":[0.0,1.0,2.0,2.5,3.0]}' -l ep5g/prepped_loc_val -n X,Y
python -m mixturemodels prep_dataset -d ep5g/measurement_validation -x '{"X":[0.0,0.5,1.0,2.0,3.0,4.0],"Y":[0.0,1.0,2.0,2.5,3.0]}' -l ep5g/prepped_loc_comp -n X,Y -s 9800
python -m mixturemodels prep_dataset -d ep5g/measurement_validation -l ep5g/prepped_nocond
```

Train models without noise on two cases: first without including any validation point samples, second including validation point samples (limited)
```
python -m mixturemodels train -d ep5g/prepped_loc -l ep5g/trained_loc_send -c mixturemodels/ep5g/train_conf_ul.json -e 9
python -m mixturemodels train -d ep5g/prepped_loc_comp -l ep5g/trained_loc_send_comp -c mixturemodels/ep5g/train_conf_ul.json -e 9
```

Train models with small noise (1ms) on two cases: first without including any validation point samples, second including validation point samples (limited)
```
python -m mixturemodels train -d ep5g/prepped_loc -l ep5g/trained_loc_send_noisy -c mixturemodels/ep5g/train_conf_ul_noisy.json -e 9
python -m mixturemodels train -d ep5g/prepped_loc_comp -l ep5g/trained_loc_send_comp_noisy -c mixturemodels/ep5g/train_conf_ul_noisy.json -e 9
```

Note that we use all the training samples in this case.

Run evaluation and plot the results. We train 2 sets of models. First, the ones that do not include the validation points samples (`trained_loc_send`), second, the models that were trained with samples that included validation points (`trained_loc_send_comp`).
```
python -m mixturemodels evaluate_pred -d ep5g/prepped_loc_val -t send -x 4,12,21 -m ep5g/trained_loc_send.gmevm,ep5g/trained_loc_send_comp.gmevm,ep5g/trained_loc_send.gmm,ep5g/trained_loc_send_comp.gmm -l ep5g/evaluate_pred_loc_agg -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_pred_loc_agg -m ep5g/trained_loc_send.gmevm,ep5g/trained_loc_send.gmm,ep5g/trained_loc_send_comp.gmevm,ep5g/trained_loc_send_comp.gmm -x 0,1,2 -y 5,25 -t tail -u 1e-6,1
```

Noisy non-complete:
```
python -m mixturemodels train -d ep5g/prepped_loc -l ep5g/trained_loc_send_noisy -c mixturemodels/ep5g/train_conf_ul_noisy.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_loc_val -t send -x 4,12,21 -m ep5g/trained_loc_send_noisy.gmevm,ep5g/trained_loc_send_noisy.gmm -l ep5g/evaluate_pred_loc_noisy_agg -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_pred_loc_noisy_agg -m ep5g/trained_loc_send_noisy.gmevm,ep5g/trained_loc_send_noisy.gmm -x 0,1,2 -y 5,25 -t tail -u 1e-6,1
```

Noisy complete:
```
python -m mixturemodels train -d ep5g/prepped_loc_comp -l ep5g/trained_loc_send_comp_noisy -c mixturemodels/ep5g/train_conf_ul_noisy.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_loc_val -t send -x 4,12,21 -m ep5g/trained_loc_send_comp_noisy.gmevm,ep5g/trained_loc_send_comp_noisy.gmm -l ep5g/evaluate_pred_loc_comp_noisy_agg -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_pred_loc_comp_noisy_agg -m ep5g/trained_loc_send_comp_noisy.gmevm,ep5g/trained_loc_send_comp_noisy.gmm -x 0,1,2 -y 5,25 -t tail -u 1e-6,1
```


Evaluate non-conditional distributions without neural network (NN), with large number of samples, without noise (FINAL)
```
python -m mixturemodels train -d ep5g/prepped_nocond -l ep5g/trained_send_l_nocond -c mixturemodels/ep5g/train_conf_ul_l_nocond.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_nocond -t send -m ep5g/trained_send_l_nocond.gmevm,ep5g/trained_send_l_nocond.gmm -l ep5g/evaluate_send_l_nocond -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_l_nocond -m ep5g/trained_send_l_nocond.gmevm,ep5g/trained_send_l_nocond.gmm -y 5,30 -t tail -u 1e-6,1
```

Evaluate non-conditional distributions without neural network (NN), with medium number of samples, without noise (FINAL)
```
python -m mixturemodels train -d ep5g/prepped_nocond -l ep5g/trained_send_m_nocond -c mixturemodels/ep5g/train_conf_ul_m_nocond.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_nocond -t send -m ep5g/trained_send_m_nocond.gmevm,ep5g/trained_send_m_nocond.gmm -l ep5g/evaluate_send_m_nocond -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_m_nocond -m ep5g/trained_send_m_nocond.gmevm,ep5g/trained_send_m_nocond.gmm -y 5,30 -t tail -u 1e-6,1
```

Evaluate non-conditional distributions without neural network (NN), with small number of samples, without noise (FINAL)
```
python -m mixturemodels train -d ep5g/prepped_nocond -l ep5g/trained_send_s_nocond -c mixturemodels/ep5g/train_conf_ul_s_nocond.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_nocond -t send -m ep5g/trained_send_s_nocond.gmevm,ep5g/trained_send_s_nocond.gmm -l ep5g/evaluate_send_s_nocond -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_s_nocond -m ep5g/trained_send_s_nocond.gmevm,ep5g/trained_send_s_nocond.gmm -y 5,30 -t tail -u 1e-6,1
```

Evaluate non-conditional distributions without neural network (NN), with very small number of samples, without noise (FINAL)
```
python -m mixturemodels train -d ep5g/prepped_nocond -l ep5g/trained_send_xs_nocond -c mixturemodels/ep5g/train_conf_ul_xs_nocond.json -e 9
python -m mixturemodels evaluate_pred -d ep5g/prepped_nocond -t send -m ep5g/trained_send_xs_nocond.gmevm,ep5g/trained_send_xs_nocond.gmm -l ep5g/evaluate_send_xs_nocond -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_xs_nocond -m ep5g/trained_send_xs_nocond.gmevm,ep5g/trained_send_xs_nocond.gmm -y 5,30 -t tail -u 1e-6,1
```

Plot aggregate 
```
python -m mixturemodels evaluate_pred -d ep5g/prepped_nocond -t send -m ep5g/trained_send_m_nocond.gmevm,ep5g/trained_send_s_nocond.gmevm,ep5g/trained_send_xs_nocond.gmevm,ep5g/trained_send_m_nocond.gmm,ep5g/trained_send_s_nocond.gmm,ep5g/trained_send_xs_nocond.gmm -l ep5g/evaluate_send_agg_nocond -y 0,300,30
python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_agg_nocond -m ep5g/trained_send_m_nocond.gmevm,ep5g/trained_send_xs_nocond.gmevm,ep5g/trained_send_m_nocond.gmm,ep5g/trained_send_xs_nocond.gmm -y 5,30 -t tail -u 1e-6,1
```

Run 3d contour plot. Set `target_delay` parameter in the script to 7,10, and 13 and run:
```
python mixturemodels/ep5g/plot_3d.py
```

#### Optional steps

Optional: Validate the trained models
```
python -m mixturemodels validate_pred -d ep5g/prepped_loc -t send -x 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 -m ep5g/trained_loc_send.gmevm.0 -l validate_pred_ep5glong_loc --plot-pdf --plot-cdf -r 1 -c 3 -y 0,400,25 
python -m mixturemodels validate_pred -d ep5g/prepped_loc -t send -x 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 -m ep5g/trained_loc_send.gmevm.0 -l validate_pred_ep5glong_loc --plot-pdf --plot-cdf --plot-tail --log -r 1 -c 3 -y 0,400,30
python -m mixturemodels validate_pred -d ep5g/prepped_loc -t send -x 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 -m ep5g/trained_loc_send.gmevm.0 -l validate_pred_ep5glong_loc --plot-tail --log -r 1 -c 3 -y 0,400,25
python -m mixturemodels validate_pred -d ep5g/prepped_loc -t send -x 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 -m ep5g/trained_loc_send.gmevm.0 -l ep5g/validate_loc_send --plot-pdf --plot-cdf -r 4 -c 5 -y 0,400,30
python -m mixturemodels validate_pred -d ep5g/prepped_loc -t send -x 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 -m ep5g/trained_loc_send.gmevm.0 -l ep5g/validate_loc_send --plot-tail --log -r 4 -c 5 -y 0,400,30
```
Optional: Prep a dataset conditioned on RSRP and plot conditional datasets.
```
python -m mixturemodels prep_dataset -d ep5g/measurement_training -x '{"RSRP":[[-81,-73],[-73,-65],[-65,-57]]}' -l ep5g/prepped_rsrp -n RSRP
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_loc -x 0,4,8,12,15,19 -t rtt_scaled --plot-pdf --plot-tail --log -r 2 -c 3 -y 0,400,50 -l 1e-6,1
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_rsrp -x 0,1,2 -m .,*,2 -t send_scaled --plot-tail --log -r 0 -c 0 -y 0,400,50 -l 1e-6,1
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_rsrp -x 0,1,2 -m .,*,2 -t receive_scaled --plot-tail --log -r 0 -c 0 -y 0,400,50 -l 1e-6,1
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-cdf -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-cdf --log -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_loc -x 0,1,2 -t send_scaled --plot-tail --loglog -r 1 -c 3 -y 0,400,25
```

### WiFi Results

We focus on `wifi/measurement_length` data but we prepare both.
```
python -m mixturemodels prep_dataset -d wifi/measurement_loc --preview
python -m mixturemodels prep_dataset -d wifi/measurement_loc -x '{"X":[1,4,8],"Y":[0,5]}' -l wifi/prepped_loc -n X,Y
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_loc -x 0,1,2 -t send_scaled --plot-pdf -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_loc -x 0,1,2 -t receive_scaled --plot-pdf -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_loc -x 0,1,2 -t receive_scaled --plot-pdf --plot-tail --log -r 1 -c 3 -y 0,400,25
```
Prepare measurement_length.
```
python -m mixturemodels prep_dataset -d wifi/measurement_length --preview
python -m mixturemodels prep_dataset -d wifi/measurement_length -x '{"length":[172,3440,6880,10320]}' -l wifi/prepped_length -n length
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_length -x 0,1,2,3 -t receive_scaled --plot-pdf -r 1 -c 4 -y 0,400,100
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_length -x 0,1,2,3 -t receive_scaled --plot-pdf --plot-tail --log -r 1 -c 4 -y 0,400,100
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_length -x 0,1,2,3 -t send_scaled --plot-pdf -r 1 -c 4 -y 0,400,100
python -m mixturemodels plot_prepped_dataset -d wifi/prepped_length -x 0,1,2,3 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 4 -y 0,400,100
```

Train models on two cases: first without including any validation point samples, second including validation point samples (limited)
```
python -m mixturemodels train -d wifi/prepped_length -l wifi/trained_length_dl_l -c mixturemodels/wifi/train_conf_dl_l.json -e 9
```

```
python -m mixturemodels validate_pred -d wifi/prepped_length -t receive -x 0,1,2,3 -m wifi/trained_length_dl_l.gmevm.0,wifi/trained_length_dl_l.gmm.0 -l wifi/validate_length_l --plot-tail --log -r 2 -c 2 -y 0,400,200 
```


Non-conditional fit

prep dataset
```
python -m mixturemodels prep_dataset -d wifi/measurement_length -x '{"length":[172]}' -l wifi/prepped_nocond_length_172 -n length
python -m mixturemodels prep_dataset -d wifi/measurement_length -x '{"length":[172]}' -l wifi/prepped_nocond_length_3440 -n length
python -m mixturemodels prep_dataset -d wifi/measurement_length -x '{"length":[172]}' -l wifi/prepped_nocond_length_6880 -n length
python -m mixturemodels prep_dataset -d wifi/measurement_length -x '{"length":[172]}' -l wifi/prepped_nocond_length_10320 -n length
```

Training non-conditional no noise
```
python -m mixturemodels train -d wifi/prepped_nocond_length_172 -l wifi/trained_nocond_length_172_m -c mixturemodels/wifi/train_conf_ul_m_nocond.json -e 1
python -m mixturemodels evaluate_pred -d wifi/prepped_nocond_length_172 -t receive -m wifi/trained_nocond_length_172_m.gmevm,wifi/trained_nocond_length_172_m.gmm -l wifi/evaluate_nocond_length_172_m -y 0,300,40
python -m mixturemodels plot_evaluation -p wifi/evaluate_nocond_length_172_m -m wifi/trained_nocond_length_172_m.gmevm -y 0,40 -t tail -u 1e-6,1
```

Training non-conditional with noise
```
python -m mixturemodels train -d wifi/prepped_nocond_length_172 -l wifi/trained_nocond_length_172_m_noisy -c mixturemodels/wifi/train_conf_ul_m_noisy_nocond.json -e 9
python -m mixturemodels evaluate_pred -d wifi/prepped_nocond_length_172 -t receive -m wifi/trained_nocond_length_172_m_noisy.gmevm,wifi/trained_nocond_length_172_m_noisy.gmm -l wifi/evaluate_nocond_length_172_m_noisy -y 0,300,40
python -m mixturemodels plot_evaluation -p wifi/evaluate_nocond_length_172_m_noisy -m wifi/trained_nocond_length_172_m_noisy.gmevm,wifi/trained_nocond_length_172_m_noisy.gmm -y 0,40 -t tail -u 1e-6,1
```

python -m mixturemodels train -d wifi/prepped_nocond_length_172 -l wifi/trained_nocond_length_172_m_noisy_test -c mixturemodels/wifi/train_conf_ul_m_noisy_nocond.json -e 1
python -m mixturemodels evaluate_pred -d wifi/prepped_nocond_length_172 -t receive -m wifi/trained_nocond_length_172_m_noisy_test.gmevm -l wifi/evaluate_nocond_length_172_m_noisy_test -y 0,300,40
python -m mixturemodels plot_evaluation -p wifi/evaluate_nocond_length_172_m_noisy_test -m wifi/trained_nocond_length_172_m_noisy_test.gmevm -y 0,40 -t tail -u 1e-6,1


AppendixEVM:
```
python -m mixturemodels train_app -d wifi/prepped_nocond_length_172 -l wifi/trained_nocond_length_172_m_append -b wifi/trained_nocond_length_172_m_test.gmm.0 -c mixturemodels/wifi/train_conf_ul_m_nocond_app.json -e 9
python -m mixturemodels evaluate_pred -d wifi/prepped_nocond_length_172 -t receive -m wifi/trained_nocond_length_172_m_append.appendix -l wifi/evaluate_nocond_length_172_m_append -y 0,300,40
python -m mixturemodels plot_evaluation -p wifi/evaluate_nocond_length_172_m_append -m wifi/trained_nocond_length_172_m_append.appendix -y 0,40 -t tail -u 1e-6,1
```

```
python -m mixturemodels train_app -d wifi/prepped_nocond_length_172 -l wifi/trained_nocond_length_172_m_append_noisy -b wifi/trained_nocond_length_172_m_append.gmm.0 -c mixturemodels/wifi/train_conf_ul_m_nocond_app.json -e 9
python -m mixturemodels evaluate_pred -d wifi/prepped_nocond_length_172 -t receive -m wifi/trained_nocond_length_172_m_append_noisy.appendix -l wifi/evaluate_nocond_length_172_m_append_noisy -y 0,300,40
python -m mixturemodels plot_evaluation -p wifi/evaluate_nocond_length_172_m_append -m wifi/trained_nocond_length_172_m_append_noisy.appendix -y 0,40 -t tail -u 1e-6,1
```

### OpenAirInterface 5G Results

Download the parquet files, store them in `mixturemodels/oai5g/measurement_results`.

We analyze uplink delays, conditioned on the uplink MCS indices: 3,5, and 7. It contains 5M samples in total.
```
python -m mixturemodels prep_dataset -d oai5g/measurement -x '{"mcs_UL":[3,5,7]}' -l oai5g/prepped_ulmcs_norm_no1 -n mcs_UL
python -m mixturemodels plot_prepped_dataset -d oai5g/prepped_ulmcs_norm_no1 -x 0,1,2 -m .,*,2 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 3 -y 0,400,50 -l 1e-6,1
python -m mixturemodels plot_prepped_dataset -d oai5g/prepped_ulmcs_norm_no1 -x 0,1,2 -m .,*,2 -t send_scaled --plot-tail --log -r 0 -c 0 -y 0,400,50 -l 1e-6,1
```

Create another dataset, without MCS=5. This dataset is used for evaluation of the ML model's generalization.
```
python -m mixturemodels prep_dataset -d oai5g/measurement -x '{"mcs_UL":[3,7]}' -l oai5g/prepped_ulmcs_norm_no1a5 -n mcs_UL
```


Default training config:
```
{
    "gmevm" : {
        "type": "gmevm",
        "bayesian": false,
        "centers": 15,
        "hidden_sizes": [10, 100, 100, 80],
        "condition_labels" : ["mcs_UL_normed"],
        "y_label" : "send_noisy",
        "training_params": {
            "dataset_size": 1000000,
            "batch_size": 128000,
            "rounds" : [
                {"learning_rate": 1e-2, "epochs":200},
                {"learning_rate": 1e-3, "epochs":200},
                {"learning_rate": 1e-4, "epochs":200},
                {"learning_rate": 1e-5, "epochs":200}
            ]
        }
    }
}
```

Train predictors on 3 levels of training samples with Gaussian noise. Large: 1M samples (20%), Medium: 256k samples (5%), Small: 64k samples (1.3%). Then compare them. Batch size is always 1/8 of the training samples.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l_noisy -c mixturemodels/oai5g/train_conf_l_noisy.json -e 5
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_noisy.gmm,oai5g/trained_ulmcs_l_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_l_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_noisy -m oai5g/trained_ulmcs_l_noisy.gmevm,oai5g/trained_ulmcs_l_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single

python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_m_noisy -c mixturemodels/oai5g/train_conf_m_noisy.json -e 9
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_m_noisy.gmm,oai5g/trained_ulmcs_m_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_m_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_noisy -m oai5g/trained_ulmcs_m_noisy.gmevm,oai5g/trained_ulmcs_m_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single

python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_s_noisy -c mixturemodels/oai5g/train_conf_s_noisy.json -e 9
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_s_noisy.gmm,oai5g/trained_ulmcs_s_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_s_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_s_noisy -m oai5g/trained_ulmcs_s_noisy.gmevm,oai5g/trained_ulmcs_s_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```


Train predictors without additional white Gaussian noise with the large number of samples. In training config, set `y_label` to `send_standard`.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l -c mixturemodels/oai5g/train_conf_l.json -e 5
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l.gmm,oai5g/trained_ulmcs_l.gmevm -l oai5g/evaluate_pred_ulmcs_l -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l -m oai5g/trained_ulmcs_l.gmevm,oai5g/trained_ulmcs_l.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```

Train predictors with additional white Gaussian noise with variance 3ms with the medium number of samples. In training config, set `y_label` to `send_verynoisy`.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_m_verynoisy -c mixturemodels/oai5g/train_conf_m_verynoisy.json -e 9
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_m_verynoisy.gmm,oai5g/trained_ulmcs_m_verynoisy.gmevm -l oai5g/evaluate_pred_ulmcs_m_verynoisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_verynoisy -m oai5g/trained_ulmcs_m_verynoisy.gmevm,oai5g/trained_ulmcs_m_verynoisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```


Train predictors without MCS=5.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1a5 -l oai5g/trained_ulmcs_l_no5 -c mixturemodels/oai5g/train_conf_l_noisy_no5.json -e 7
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_no5.gmm,oai5g/trained_ulmcs_l_no5.gmevm -l oai5g/evaluate_pred_ulmcs_l_no5 -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_no5 -m oai5g/evaluate_pred_ulmcs_l_no5.gmevm,oai5g/evaluate_pred_ulmcs_l_no5.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```

Train predictors without MCS=5 very noisy.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1a5 -l oai5g/trained_ulmcs_l_verynoisy_no5 -c mixturemodels/oai5g/train_conf_l_verynoisy_no5.json -e 6
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_verynoisy_no5.gmm,oai5g/trained_ulmcs_l_verynoisy_no5.gmevm -l oai5g/evaluate_pred_ulmcs_l_verynoisy_no5 -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_verynoisy_no5 -m oai5g/trained_ulmcs_l_verynoisy_no5.gmevm,oai5g/trained_ulmcs_l_verynoisy_no5.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```

# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

