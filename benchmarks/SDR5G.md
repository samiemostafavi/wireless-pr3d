# SDR5G

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