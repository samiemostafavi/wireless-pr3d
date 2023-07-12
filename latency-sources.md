# Latency Sources

## Wireless Link

### A) Queuing or Buffering
* Related to the load
* Packet scheduling and fragmentation

### B) Scheduling Request
*  Release 15 is there, URLLC likely not. Do we have that?

### C) Waveform Processing
* Encoding, decoding
* Modulation

### D) Resource allocation
* Resource block allocation
* Transmit power allocation

### E) Retransmissions (HARQ scheme)
* Modulation and coding scheme index (MSC) chosen from channel quality (CQI) to maximize the data throughput while maintaining an acceptable error rate.

## Testbed Connections, OS Networking stack, etc
* Linux networking (~100us)
* End-node - nrUE 10Gbps link (~50us)
* Server - 5G gateway 10 Gbps link (~50us)
* Switching delay (~10us)

In total far less than 1ms on average.

These are averages, how about the latency outliers? can it get to a few milliseconds?
