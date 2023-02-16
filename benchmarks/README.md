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
pip install -Ur requirements.txt
```

### Mixture Models

Open all the `parquet` files and show a preview of them:
```
python -m mixturemodels prep_dataset -d ep5g --preview
```

Prepare the dataset by the conditions defined in json format with `-x` argument.
```
python -m mixturemodels prep_dataset -d ep5g -x '{"X":[1,2],"Y":[1,2,3],"RSRP":[[-75.0,-58.0]]}' -l prepped_ep5g
```

You can normalize certain columns using `-n`:
```
python -m mixturemodels prep_dataset -d ep5g -x '{"X":[1,2],"Y":[1,2,3],"RSRP":[[-75.0,-58.0]]}' -l prepped_ep5g -n rtt
```

Prepare the dataset by the conditions defined in json format with `-x` argument and plot the pdf and cdf for each dataframe.
```
python -m mixturemodels prep_dataset -d ep5g -x '{"X":[1,2],"Y":[1,2,3],"RSRP":[[-75.0,-58.0]]}' -l prepped_ep5g -t rtt -r 1 -c 3 -y 8536766,1000,30452346 --plot-cdf --plot-pdf
```

Train models
```
python -m mixturemodels train -d prepped_ep5g_norm -l trained_ep5g -c mixturemodels/train_conf.json -e 10
```

Validate the models
```
python -m mixturemodels validate_pred -d prepped_ep5g_norm -t rtt_scaled -x 0,1,2 -m trained_ep5g.gmm.1,trained_ep5g.gmevm.1 -l validate_pred_ep5g -r 1 -c 3 -y 0,1000,1
```

Plot the results (NOT IMPLEMENTED)
```
```


# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

