# WiFi Benchmarking Schemes


Location-based measurements:

- Number of recorded samples per run: 1e6

| Run #        |  X  |  Y  |  RSSI  | ul capacity | dl capacity | load packet len | load interval |
| -----------  | --- | --- | ------ | ----------- | ----------- | --------------- | ------------- |
| 1            |  1  |  0  | -61dbm | 17.93Mbps   | 26.22Mbps   | 172B            | 10ms          |
| 2            |  4  |  5  | -77dbm | 16.68Mbps   | 26.22Mbps   | 172B            | 10ms          |
| 3            |  8  |  5  | -87dbm | 9.67Mbps    | 9.26Mbps    | 172B            | 10ms          |


Packet length measurements:

| Run #        |  X  |  Y  |  RSSI  | ul capacity | dl capacity | load packet len | load interval | util % |
| -----------  | --- | --- | ------ | ----------- | ----------- | --------------- | ------------- | ------ |
| 4            |  1  |  0  | -61dbm | 17.93Mbps   | 26.22Mbps   | 172B            | 10ms          | 0.76%  |
| 5            |  1  |  0  | -61dbm | 17.93Mbps   | 26.22Mbps   | 3440B           | 10ms          | 15.34% |
| 6            |  1  |  0  | -61dbm | 17.93Mbps   | 26.22Mbps   | 6880B           | 10ms          | 30.69% |
| 7            |  1  |  0  | -61dbm | 17.93Mbps   | 26.22Mbps   | 10320B          | 10ms          | 46.04% |
