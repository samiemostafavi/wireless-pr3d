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

# Access the Measurements Datasets

All the measurements are published on Kaggle: [wireless-pr3d](https://www.kaggle.com/datasets/samiemostafavi/wireless-pr3d).

You can download them if you install Kaggle Python package and add your token
```
pip install kaggle
vim /home/wlab/.kaggle/kaggle.json
```

Then download the dataset by running
```
cd benchmarks
kaggle datasets download -d samiemostafavi/wireless-pr3d
unzip wireless-pr3d.zip

Prepare COTS 5G dataset:
```
mv COTS-5G ./mixturemodels/ep5g/measurement_validation_results
```

Prepare SDR 5G dataset:
```
mv SDR-5G ./mixturemodels/oai5g/measurement_results
```

Prepare IEEE802.11g dataset: 
```
mkdir ./mixturemodels/wifi/measurement_loc_results
find ./IEEE80211g -type f -name 'dataset_*_172*' -exec mv -t ./mixturemodels/wifi/measurement_loc_results/ {} +
mv IEEE80211g/ ./mixturemodels/wifi/measurement_length_results
for i in {0..5}; do
  find ./mixturemodels/wifi/measurement_loc_results/ -type f -name "dataset_${i}_172*" -exec mv -t ./mixturemodels/wifi/measurement_length_results/ {} +
  find test -type f -name "dataset_${i}_172*" -exec cp -t test2 {} +
done
```


## Time Agnostic Analysis

You can find the scripts to run the benchmarks in:
1. [COTS 5G](./mixturemodels/COTS5G.md)
2. [IEEE802.11g](./mixturemodels/IEEE80211g.md)
3. [SDR 5G](./mixturemodels/SDR5G.md)

## Time-Series Forcasting Analysis



# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

