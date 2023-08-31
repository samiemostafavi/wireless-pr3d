# Figure 1
This figure works with "old_plot_evaluation.py"

- mixturemodels/ep5g/prepped_nocond_results/receive_scaled_log_tail

`python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-pdf -r 0 -c 0 -y 3,300,30`

- mixturemodels/ep5g/prepped_nocond_results/receive_scaled_pdf

`python -m mixturemodels plot_prepped_dataset -d ep5g/prepped_nocond -t receive_scaled -x 0 -m . --plot-tail --log -r 0 -c 0 -y 3,300,30`

# Figure 4
This figure works with "old_plot_evaluation.py"

- mixturemodels/ep5g/evaluate_send_agg_nocond_results

`python -m mixturemodels plot_evaluation -p ep5g/evaluate_send_agg_nocond -m ep5g/trained_send_m_nocond.gmevm,ep5g/trained_send_xs_nocond.gmevm,ep5g/trained_send_m_nocond.gmm,ep5g/trained_send_xs_nocond.gmm -y 5,30 -t tail -u 1e-6,1`

# Figure 5
- mixturemodels/oai5g/evaluate_pred_ulmcs_l_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l -m oai5g/trained_ulmcs_l.gmevm,oai5g/trained_ulmcs_l.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`

# Figure 6

## A
- mixturemodels/oai5g/evaluate_pred_ulmcs_l_noisy_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_noisy -m oai5g/trained_ulmcs_l_noisy.gmevm,oai5g/trained_ulmcs_l_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`

## B
- mixturemodels/oai5g/evaluate_pred_ulmcs_m_noisy_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_noisy -m oai5g/trained_ulmcs_m_noisy.gmevm,oai5g/trained_ulmcs_m_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`

## C
- mixturemodels/oai5g/evaluate_pred_ulmcs_s_noisy_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_s_noisy -m oai5g/trained_ulmcs_s_noisy.gmevm,oai5g/trained_ulmcs_s_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`

# Figure 7
This figure works with "old_plot_evaluation.py"

- mixturemodels/oai5g/evaluate_pred_ulmcs_m_verynoisy_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_verynoisy -m oai5g/trained_ulmcs_m_verynoisy.gmevm,oai5g/trained_ulmcs_m_verynoisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`

# Figure 8
- mixturemodels/oai5g/evaluate_pred_ulmcs_l_no5_results

`python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_no5 -m oai5g/evaluate_pred_ulmcs_l_no5.gmevm,oai5g/evaluate_pred_ulmcs_l_no5.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single`