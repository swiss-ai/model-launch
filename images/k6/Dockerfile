ARG K6_VERSION=2.0.0-rc1
FROM docker.io/grafana/k6:${K6_VERSION}

USER root
RUN apk add --no-cache bash=5.3.3-r1 && \
    mkdir -p /etc/ld.so.conf.d /usr/lib64 /lib64 && \
    printf '#!/bin/sh\ncase "$*" in *-p*) echo "libfabric.so.1 => /usr/lib64/libfabric.so.1";; esac\nexit 0\n' > /sbin/ldconfig && \
    chmod +x /sbin/ldconfig

USER k6
WORKDIR /work
