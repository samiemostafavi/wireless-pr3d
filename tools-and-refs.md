# Tools 


## Measurements

### SDR 5G

Core network:

OAI CN5G pre-requisites:

1) Install docker-compose if not already installed:

# https://docs.docker.com/compose/install/
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

2) OAI CN5G Setup

# Git oai-cn5g-fed repository
git clone https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git ~/oai-cn5g-fed

# Pull docker images
docker pull oaisoftwarealliance/oai-amf:develop
docker pull oaisoftwarealliance/oai-nrf:develop
docker pull oaisoftwarealliance/oai-smf:develop
docker pull oaisoftwarealliance/oai-udr:develop
docker pull oaisoftwarealliance/oai-udm:develop
docker pull oaisoftwarealliance/oai-ausf:develop
docker pull oaisoftwarealliance/oai-spgwu-tiny:develop
docker pull oaisoftwarealliance/trf-gen-cn5g:latest

# Tag docker images
docker image tag oaisoftwarealliance/oai-amf:develop oai-amf:develop
docker image tag oaisoftwarealliance/oai-nrf:develop oai-nrf:develop
docker image tag oaisoftwarealliance/oai-smf:develop oai-smf:develop
docker image tag oaisoftwarealliance/oai-udr:develop oai-udr:develop
docker image tag oaisoftwarealliance/oai-udm:develop oai-udm:develop
docker image tag oaisoftwarealliance/oai-ausf:develop oai-ausf:develop
docker image tag oaisoftwarealliance/oai-spgwu-tiny:develop oai-spgwu-tiny:develop
docker image tag oaisoftwarealliance/trf-gen-cn5g:latest trf-gen-cn5g:latest


3) OAI CN5G Configuration files

Download and copy configuration files:

Copy docker-compose-basic-nrf.yaml to ~/oai-cn5g-fed/docker-compose

wget -O ~/oai-cn5g-fed/docker-compose/docker-compose-basic-nrf.yaml https://gitlab.eurecom.fr/oai/openairinterface5g/-/raw/develop/doc/tutorial_resources/docker-compose-basic-nrf.yaml?inline=false


Copy oai_db.sql to ~/oai-cn5g-fed/docker-compose/database

wget -O ~/oai-cn5g-fed/docker-compose/database/oai_db.sql https://gitlab.eurecom.fr/oai/openairinterface5g/-/raw/develop/doc/tutorial_resources/oai_db.sql?inline=false


Run OAI CN5G

cd ~/oai-cn5g-fed/docker-compose
python3 core-network.py --type start-basic --scenario 1


gnodeb:

```
COMMANDS
```
OAI gNB pre-requisites

1) Build UHD from source

sudo apt install -y libboost-all-dev libusb-1.0-0-dev doxygen python3-docutils python3-mako python3-numpy python3-requests python3-ruamel.yaml python3-setuptools cmake build-essential

git clone https://github.com/EttusResearch/uhd.git ~/uhd
cd ~/uhd
git checkout v4.3.0.0
cd host
mkdir build
cd build
cmake ../
make -j $(nproc)
make test # This step is optional
sudo make install
sudo ldconfig
sudo uhd_images_downloader


2) Build OAI gNB

# Get openairinterface5g source code
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git ~/openairinterface5g
cd ~/openairinterface5g
git checkout develop

# Install OAI dependencies
cd ~/openairinterface5g
source oaienv
cd cmake_targets
./build_oai -I

# Build OAI gNB
cd ~/openairinterface5g
source oaienv
cd cmake_targets
./build_oai -w USRP --ninja --nrUE --gNB --build-lib all -c

3) RUN OAI gNB

cd ~/openairinterface5g
source oaienv
cd cmake_targets/ran_build/build
sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --sa -E --continuous-tx



nrUE:
```
COMMANDS
```
RUN OAI nrUE

cd ~/openairinterface5g
source oaienv
cd cmake_targets/ran_build/build
sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --nokrnmod --ue-fo-compensation --sa -E --uicc0.imsi 001010000000001 --uicc0.nssai_sd 1


Refs:
- https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_CN5G_gNB_USRP_COTS_UE_Tutorial.md
- ?
- https://github.com/KTH-EXPECA/oai5g-docker

SDR LTE: ?

WiFi: ?

Data prepreocessig: https://github.com/KTH-EXPECA/measurements-service
Collecting network parameters: https://github.com/KTH-EXPECA/advantech-networkinfo-daemon

## Training and Evaluation

PR3D: Deep learning-based conditional delay predictors
