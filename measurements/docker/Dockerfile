FROM golang:1.19.3
RUN go install github.com/samiemostafavi/nlmt/cmd/irtt@latest

RUN apt-get update
RUN apt-get install -y iperf3
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip
RUN pip install keystoneauth1
RUN pip install python-swiftclient

RUN mkdir -p /tmp/results
WORKDIR /tmp/results

COPY entrypoint.sh /tmp/entrypoint.sh
COPY upload-files.py /tmp/upload-files.py
COPY make-parquet.py /tmp/make-parquet.py
COPY adv-mobile-info-recorder.py /tmp/adv-mobile-info-recorder.py
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
