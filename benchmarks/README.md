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
pip install -Ur requirements.txt
```

To use SciencePlot, latex must be installed on your machine
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

### Commercial 5G

Open all the `parquet` files and show a preview of them:
```
python -m mixturemodels prep_dataset -d ep5g --preview
python -m mixturemodels prep_dataset -d ep5glong --preview
```

Prepare the dataset by the conditions defined in json format with `-x` argument.
```
python -m mixturemodels prep_dataset -d ep5glong -x '{"X":[0.5,2.0,4.0],"Y":[0.0,2.5]}' -l prepped_ep5glong_loc
python -m mixturemodels prep_dataset -d ep5glong -x '{"RSRP":[[-81,-73],[-73,-65],[-65,-57]]}' -l prepped_ep5glong_rsrp
```

You can normalize features using `-n`:
```
python -m mixturemodels prep_dataset -d ep5glong -x '{"X":[0.5,2.0,4.0],"Y":[0.0,2.5]}' -l prepped_ep5glong_loc_norm -n X,Y
```

Plot the pdf, cdf, and tail plots for the prepared conditional dataframes.
```
python -m mixturemodels plot_prepped_dataset -d prepped_ep5glong_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-cdf -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d prepped_ep5glong_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-cdf --log -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d prepped_ep5glong_loc -x 0,1,2 -t send_scaled --plot-tail --loglog -r 1 -c 3 -y 0,400,25
```

Train models
```
python -m mixturemodels train -d prepped_ep5glong_loc_norm -l trained_ep5glong_loc_send -c mixturemodels/train_conf_ul.json -e 9
```

Validate the trained models
```
python -m mixturemodels validate_pred -d prepped_ep5glong_loc -t send -x 0,1,2 -m trained_ep5glong_loc_send.gmm.0 -l validate_pred_ep5glong_loc --plot-pdf --plot-cdf -r 1 -c 3 -y 0,400,25 
python -m mixturemodels validate_pred -d prepped_ep5glong_loc -t send -x 0,1,2 -m trained_ep5glong_loc_send.gmm.0 -l validate_pred_ep5glong_loc --plot-pdf --plot-cdf --plot-tail --log -r 1 -c 3 -y 0,400,30
python -m mixturemodels validate_pred -d prepped_ep5glong_loc -t send -x 0,1,2 -m trained_ep5glong_loc_send.gmm.0 -l validate_pred_ep5glong_loc --plot-tail --loglog -r 1 -c 3 -y 0,400,25
```

Run evaluation and plot the results
```
python -m mixturemodels evaluate_pred -d prepped_ep5glong_loc_norm -t send -x 0,1,2 -m trained_ep5glong_loc_send.gmm,trained_ep5glong_loc_send.gmevm -l evaluate_pred_ep5glong_loc -y 0,300,25
python -m mixturemodels plot_evaluation -p evaluate_pred_ep5glong_loc -m gmm,gmevm -x 0,1,2 -y 0,25
```


### IEEE802.11n

```
python -m mixturemodels prep_dataset -d wifi_loc --preview
python -m mixturemodels prep_dataset -d wifi_loc -x '{"X":[1,4,8],"Y":[0,5]}' -l prepped_wifi_loc -n X,Y
python -m mixturemodels plot_prepped_dataset -d prepped_wifi_loc -x 0,1,2 -t send_scaled --plot-pdf -r 1 -c 3 -y 0,400,25
python -m mixturemodels plot_prepped_dataset -d prepped_wifi_loc -x 0,1,2 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 3 -y 0,400,25
```

```
python -m mixturemodels prep_dataset -d wifi_length -x '{"length":[172,3440,6880,10320]}' -l prepped_wifi_length -n length
python -m mixturemodels plot_prepped_dataset -d prepped_wifi_length -x 0,1,2,3 -t send_scaled --plot-pdf -r 1 -c 4 -y 0,400,100
python -m mixturemodels plot_prepped_dataset -d prepped_wifi_length -x 0,1,2,3 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 4 -y 0,400,100
```

# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

