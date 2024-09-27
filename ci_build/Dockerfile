FROM ubuntu:24.02

# sudo isn't really needed, but we include it for convenience
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget build-essential rsync sudo python3 python3-dev python3-distutils python-is-python3 git

# Install the latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# Install pip packages
RUN python3 -m pip install --upgrade numpy

# Change work directory
WORKDIR /workspace



