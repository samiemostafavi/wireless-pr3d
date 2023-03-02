# SDR 5G Measurement Commands

## 1) Start core network

Forlong (cn):

```
```

## 2) Start gnodeb

Finarfin (gnodeb):

```
sudo docker run --rm --privileged --network host -e USE_SA_TDD_MONO_E320='yes' -e GNB_ID='e00' -e GNB_NAME='gNBOAI' -e MCC='001' -e MNC='01' -e MNC_LENGTH='2' -e TAC='1' -e NSSAI_SST='1' -e NSSAI_SD='1' -e AMF_IP_ADDRESS='192.168.70.132' -e GNB_NGA_IF_NAME='enp5s0' -e GNB_NGA_IP_ADDRESS='192.168.70.139' -e GNB_NGU_IF_NAME='enp5s0' -e GNB_NGU_IP_ADDRESS='192.168.70.139' -e ATT_TX='0' -e ATT_RX='0' -e MAX_RXGAIN='114' -e SDR_ADDRS='addr=10.31.1.202' -e THREAD_PARALLEL_CONFIG='PARALLEL_SINGLE_THREAD' -e USE_ADDITIONAL_OPTIONS='--sa --usrp-tx-thread-config 1 -E --gNBs.[0].min_rxtxtime 6' --name 5g-gnodeb samiemostafavi/expeca-oai-gnb
```

Limit mcs to 1:
```
sudo docker run --rm --privileged --network host -e USE_SA_TDD_MONO_E320='yes' -e GNB_ID='e00' -e GNB_NAME='gNBOAI' -e MCC='001' -e MNC='01' -e MNC_LENGTH='2' -e TAC='1' -e NSSAI_SST='1' -e NSSAI_SD='1' -e AMF_IP_ADDRESS='192.168.70.132' -e GNB_NGA_IF_NAME='enp5s0' -e GNB_NGA_IP_ADDRESS='192.168.70.139' -e GNB_NGU_IF_NAME='enp5s0' -e GNB_NGU_IP_ADDRESS='192.168.70.139' -e ATT_TX='0' -e ATT_RX='0' -e MAX_RXGAIN='114' -e SDR_ADDRS='addr=10.31.1.202' -e THREAD_PARALLEL_CONFIG='PARALLEL_SINGLE_THREAD' -e USE_ADDITIONAL_OPTIONS='--sa --usrp-tx-thread-config 1 -E --gNBs.[0].min_rxtxtime 6 --MACRLCs.[0].dl_max_mcs 1 --MACRLCs.[0].ul_max_mcs 1' --name 5g-gnodeb samiemostafavi/expeca-oai-gnb
```

## 3) Start nrue

Fingolfin (nrue):

```
docker run --rm --privileged --network host -e FULL_IMSI='001010000000001' -e FULL_KEY='fec86ba6eb707ed08905757b1bb44b8f' -e OPC='C42449363BBAD02B66D16BC975D77CC1' -e DNN='oai' -e NSSAI_SST='1' -e USE_ADDITIONAL_OPTIONS='-r 106 --numerology 1 --band 78 -C 3619200000 --nokrnmod --sa -E --uicc0.imsi 001010000000001 --uicc0.nssai_sd 1 --usrp-args addr=10.31.1.201 --ue-fo-compensation --ue-rxgain 120 --ue-txgain 0 --ue-max-power 0' --name 5g-nrue samiemostafavi/expeca-oai-nr-ue
```

## 4) Test bandwidth

Make sure `iperf3 -s` is running both on Fingolfin and spgwu container at Forlong.

Uplink at Fingolfin (nrue):
```
iperf3 -c 12.1.1.1 -u -b 100M --get-server-output
```

Downlink at Forlong (cn):
```
docker exec 5gcn-7-spgwu-irtt iperf3 -c 12.1.1.19 -u -b 100M --get-server-output
```

## 5) Run irtt

Forlong (cn):
```
docker exec -d 5gcn-7-spgwu-irtt irtt server -i 0 -d 0 -l 0
```

Fingolfin (nrue):
```
irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_0.json --fill=rand 12.1.1.1
```


# SDR 5G Measurement Schemes


## Delay profile relation to MCS index

### Measurement 1:

Average RSRP: -112

5G Uplink:
- MCS: 7
- Bandwidth: 6.0Mbps

5G Downlink:
- MCS: 7
- Bandwidth: 14.0Mbps

Uplink load:
- Packet length: 75Bytes 
- Packet interval: 10ms
- Bitrate: 60kbps 
- Utilization: 1%

Downlink load:
- Packet length: 75Bytes
- Packet interval: 10ms
- Bitrate: 60kbps 
- Utilization: 0.42%

