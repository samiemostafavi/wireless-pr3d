[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Start

Create a Python 3.9 virtual environment using `virtualenv`

```
cd benchmarks
python -m virtualenv --python=python3.9 ./venv
source venv/bin/activate
```

To use pySpark Java development kit (JDK) must be installed on your machine
```
sudo apt-get install openjdk-8-jdk
```

Install python dependencies
```
pip install -Ur requirements.txt
```

To use SciencePlot, latex must be installed on your machine
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

# COTS 5G



# IEEE802.11g



# SDR 5G



# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

