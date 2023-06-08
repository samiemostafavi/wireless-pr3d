# COTS5G

Download the parquet files, store them in `mixturemodels/ep5g/measurement_training_results` and `mixturemodels/ep5g/measurement_validation_results`.

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

## Optional steps

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

First figure of the paper, downlink latency measurements
```
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-pdf -r 0 -c 0 -y 3,300,30
python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-tail --log -r 0 -c 0 -y 3,300,30
```
