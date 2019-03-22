FROM ubuntu
MAINTAINER leo.cazenille@gmail.com

RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq gosu python3-pip python3-yaml git openssh-server openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 --no-cache-dir install numpy scoop

RUN mkdir -p /home/user

ENTRYPOINT ["/home/user/shoal-cluster-dockers/python-pi/entrypoint.sh"]

RUN git clone https://gitlab.com/leo.cazenille/shoal-cluster-dockers.git /home/user/shoal-cluster-dockers

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
