FROM ubuntu
MAINTAINER leo.cazenille@gmail.com

RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq gosu python3-pip python3-yaml git openssh-server openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 --no-cache-dir install numpy scoop scipy cma

RUN mkdir -p /home/user

ENTRYPOINT ["/home/user/Badoit/entrypoint.sh"]

RUN git clone https://github.com/jpbruneton/Badoit.git /home/user/Badoit

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
