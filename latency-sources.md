# Latency Sources

## Wireless Link

### A) Queuing or Buffering
* Related to the load
* Packet scheduling and fragmentation

### B) Wireless Transmission Processes
* Encoding, decoding
* Modulation

### C) Resource allocation
* Resource block allocation
* Transmit power allocation

### D) Retransmissions (HARQ scheme)
* Modulation and conding scheme index chosen from channel quality to keep the error rate low?
* Dynamic

## Testbed Connections, OS Networking stack, etc
* Linux networking (~100us)
* End-node - nrUE 10Gbps link (~50us)
* Server - 5G gateway 10 Gbps link (~50us)
* Switching delay (~10us)

In total far less than 1ms on average.

These are averages, how about the latency outliers? can it get to a few milliseconds?
