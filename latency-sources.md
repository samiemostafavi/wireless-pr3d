# Latency Sources

## Wireless Link

### A) Queuing or Buffering
* Related to the load
* Packet scheduling and fragmentation

### B) Scheduling Request
* Uplink:
    * Dynamic Scheduling
    * CS(Configured Scheduling)
* Downlink:
    * Dynamic Scheduling
    * SPS(Semi Persistent Scheduling)

More info: 
* [5g scheduling sharetechnote](https://www.sharetechnote.com/html/5G/5G_Scheduling.html)
* [Uplink resource allocations](https://jyos-sw.medium.com/uplink-resource-allocations-in-5g-nr-f9354e10cb6f)
  

### C) Waveform Processing
* Encoding, decoding
* Modulation

### D) Resource allocation
* Resource block allocation
* Transmit power allocation

### E) Retransmissions (HARQ scheme)
* Modulation and coding scheme index (MSC) chosen from channel quality (CQI) to maximize the data throughput while maintaining an acceptable error rate.

### F) HARQ Feedback
* Downlink ACKs: For faster feedbacks, long PUCCH or short PUCCH?

## Testbed Connections, OS Networking stack, etc
* Linux networking (~100us)
* End-node - nrUE 10Gbps link (~50us)
* Server - 5G gateway 10 Gbps link (~50us)
* Switching delay (~10us)

In total far less than 1ms on average.

These are averages, how about the latency outliers? can it get to a few milliseconds?
