FROM ubuntu:14.04
MAINTAINER Marko Kolarek <marko.kolarek@zalando.de>

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:fkrull/deadsnakes

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	python2.6 python2.6-dev \
	python2.7 python2.7-dev \
	python3.3 python3.3-dev \
	python3.4 python3.4-dev \
	python3.5 python3.5-dev

RUN apt-get install -y \
	python-pip \
	cython \
	pypy pypy-dev \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	git

RUN pip install -U pip
RUN pip install -U tox

ARG CACHEBUST=1
RUN git clone --depth=50 --branch=feature/2to3 https://github.com/zalando/expan.git

WORKDIR expan/

RUN pip install -r requirements.txt

CMD tox -e py26 py27 py33 py34 py35
