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
python -m mixturemodels prep_dataset -d ep5g -t rtt -x '{"X":[1,2],"Y":[1,2,3],"RSRP":[[-75.0,-58.0]]}' -l prepped_ep5g
```

Prepare the dataset by the conditions defined in json format with `-x` argument and plot the pdf and cdf for each dataframe.
```
python -m mixturemodels prep_dataset -d ep5g -t rtt -x '{"X":[1,2],"Y":[1,2,3],"RSRP":[[-75.0,-58.0]]}' -l prepped_ep5g -r 1 -c 3 -y 8536766,1000,30452346 --plot-cdf --plot-pdf
```

Train models
```
python -m mixturemodels train -d prepped_ep5g -l trained_ep5g -c mixturemodels/train_conf.json -e 10
```


Validate the models (NOT TESTED)
```
python -m mixturemodels validate_pred -q [0,2],[4,4],[6,6],[0,10],[8,10] -d gym_2hop_p2 -w [0.1,0.1],[0.1,0.9],[0.9,0.9] -m train_2hop_p2.gmm,train_2hop_p2.gmevm -l validate_pred_2hop_p2 -r 3 -c 5 -y 0,100,250 -e 1
```

Plot the results (NOT IMPLEMENTED)
```
```


# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

