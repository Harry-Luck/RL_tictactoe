FROM ubuntu:20.04

RUN apt-get update && \
		DEBIAN_FRONTEND="noninteractive" \
		apt-get install -y \
		pkg-config \
		vim \
		python3	\
		python3-pip && \
rm -rf /var/lib/apt/lists

RUN pip3 install \
		numpy \
		tensorflow \
		progressbar \
		pyyaml \
		h5py
