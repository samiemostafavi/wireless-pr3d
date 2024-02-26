FROM samiemostafavi/pr3d:base
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Stockholm"
RUN apt-get update -y #  && apt-get upgrade -y 
RUN pip install --upgrade pip

COPY . /pr3d_demo
WORKDIR /pr3d_demo
RUN pip install -Ur requirements.txt
