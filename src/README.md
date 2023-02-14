[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Wireless latency prediction code


## Start

Create a Python 3.9 virtual environment using `virtualenv`

        $ python -m virtualenv --python=python3.9 ./venv
        $ source venv/bin/activate

Install dependencies

        $ pip install -Ur requirements.txt


## Training and validation


Validate the dataset
```
python -m multihop_benchmark validate_gym -q [0,2],[4,4],[6,6],[0,10],[8,10] -d gym_2hop_p2 -w [0.1,0.1],[0.1,0.9],[0.9,0.9] -l validate_gym_2hop_p2 -r 3 -c 5 -y 0,100,250
python -m multihop_benchmark validate_gym -q [0],[4],[6],[8],[10] -d gym_1hop_p2 -w [0.1],[0.5],[0.9] -l validate_gym_1hop_p2 -r 3 -c 5 -y 0,100,250
```

Train multihop models
```
python -m multihop_benchmark train -d gym_2hop_p2 -l train_2hop_p2 -c multihop_benchmark/train_conf_2hop.json -e 10
```

Validate multihop models
```
python -m multihop_benchmark validate_pred -q [0,2],[4,4],[6,6],[0,10],[8,10] -d gym_2hop_p2 -w [0.1,0.1],[0.1,0.9],[0.9,0.9] -m train_2hop_p2.gmm,train_2hop_p2.gmevm -l validate_pred_2hop_p2 -r 3 -c 5 -y 0,100,250 -e 1
```

Train onehop model
```
python -m multihop_benchmark train -d gym_1hop_p2 -l train_1hop_p2 -c multihop_benchmark/train_conf_1hop.json -e 10
```

Validate onehop model
```
python -m multihop_benchmark validate_pred -q [0],[4],[6],[8],[10] -d gym_1hop_p2 -w [0.1],[0.5],[0.9] -m train_1hop_p2.gmm,train_1hop_p2.gmevm -l validate_pred_1hop_p2 -r 3 -c 5 -y 0,100,250 -e 1
```


Plot the results
```
```


# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

