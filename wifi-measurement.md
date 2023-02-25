
# Install requirements

On both Finarfin and Fingolfin hosts
```
pip3 install --upgrade setuptools pip
pip3 install paramiko
git clone https://github.com/KTH-EXPECA/sdr-tools.git
cd sdr-tools
```

# Reset the SDRs

Finarfin (AP):
```
SDR='sdr-01' JSON_PATH='sdrs.json' python reboot/reboot.py
```

Fingolfin (STA):
```
SDR='sdr-02' JSON_PATH='sdrs.json' python reboot/reboot.py
```

# Switch to WiFi

Finarfin (AP):
```
DESIGN='mango' SDR='sdr-01' JSON_PATH='sdrs.json' python3 change_design/change_design.py
```
Fingolfin (STA):
```
DESIGN='mango' SDR='sdr-02' JSON_PATH='sdrs.json' python3 change_design/change_design.py
```

# Start WiFi

Finarfin (AP):
```
DESIGN='mango' SDR='sdr-01' SIDE='ap' CONFIG='{"mac_addr":"40:d8:55:04:20:12"}' JSON_PATH='sdrs.json' python3 start_mango/start_mango.py
```

Fingolfin (STA):
```
DESIGN='mango' SDR='sdr-02' SIDE='sta' CONFIG='{"mac_addr":"40:d8:55:04:20:19"}' JSON_PATH='sdrs.json' python3 start_mango/start_mango.py
```

# Conifgure WiFi routing

Finarfin (AP):
```
SDR='sdr-01' SIDE='ap' JSON_PATH='sdrs.json' CONFIG='{"protocol":"udp","server":{"ip":"10.30.1.252","port":"50000"},"ap":{"server_port":"50500","sta_port":"50000"},"sta":{"mac_addr":"40:d8:55:04:20:19","ip":"192.168.11.10","ap_port":"50500"}}' python3 config_mango_routes/config_routes.py
```

Fingolfin (STA):
```
SDR='sdr-02' SIDE='sta' JSON_PATH='sdrs.json' CONFIG='{"protocol":"udp","client":{"ip":"10.30.1.251","port":"50000"},"sta":{"client_port":"50000","ap_port":"50500"},"ap":{"ip":"192.168.11.1","sta_port":"50000"}}' python3 config_mango_routes/config_routes.py
```

# Check WiFi link quality

Finarfin (AP):
```
SDR='sdr-01' SIDE='ap' JSON_PATH='./sdrs.json' OUTPUT_PATH='./tmp/ap_link.json' SPEED_CHECK='192.168.11.10' python check_mango/check_mango.py
```

Fingolfin (STA):
```
SDR='sdr-02' SIDE='sta' JSON_PATH='./sdrs.json' OUTPUT_PATH='./home/wlab/irtt_data/wifi/sta_link.json' SPEED_CHECK='192.168.11.1' python check_mango/check_mango.py
```
```
scp finarfin://tmp/ap_link.json /home/wlab/irtt_data/wifi/
```

# Start IRTT

Finarfin (AP):
```
docker run --rm -d --network host --name nuria-irttserver nuriafe99/irtt-server
```

Fingolfin (STA):
```
irtt client -i 10ms -d 100s -l 172 -o /home/wlab/irtt_data/wifi/rtts_0.json --fill=rand 10.30.1.3:50000 --local=0.0.0.0:50000
```

# Collect the result file

Fingolfin (STA):
```
scp fingolfin://home/wlab/irtt_data/wifi/rtts_0.json .
```
