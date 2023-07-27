FROM golang:1.19.3
RUN go install github.com/samiemostafavi/nlmt/cmd/irtt@v0.9.14

RUN apt-get update
RUN apt-get install -y iperf3
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip
RUN pip install keystoneauth1
RUN pip install python-swiftclient
RUN pip install numpy
RUN pip install pandas
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install loguru

WORKDIR /sbin/
RUN wget http://www.uftpserver.com/downloads/binaries/latest/x64/uFTP
WORKDIR /etc/
COPY uftpd.cfg /etc/uftpd.cfg
RUN chown root:root /sbin/uFTP
RUN chown root:root /etc/uftpd.cfg
RUN chmod +x /sbin/uFTP

WORKDIR /tmp/
COPY entrypoint.sh /tmp/entrypoint.sh
COPY measure-upload.sh /tmp/measure-upload.sh
COPY upload-files.py /tmp/upload-files.py
COPY makeparquet.py /tmp/makeparquet.py
COPY parquets-from-folders.py /tmp/parquets-from-folders.py
COPY adv-mobile-info-recorder.py /tmp/adv-mobile-info-recorder.py
RUN chmod +x /tmp/entrypoint.sh
RUN chmod +x /tmp/measure-upload.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
