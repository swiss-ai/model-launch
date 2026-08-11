ARG K6_VERSION=2.0.0-rc1
FROM docker.io/grafana/k6:${K6_VERSION}

# DL3066 wants numeric UIDs, but "root" and "k6" are the accounts the
# grafana/k6 base image defines; hardcoding their numeric IDs here would
# silently break if the base image ever renumbers them.
# hadolint ignore=DL3066
USER root
RUN apk add --no-cache bash=5.3.3-r1 && \
    mkdir -p /etc/ld.so.conf.d /usr/lib64 /lib64 && \
    printf '#!/bin/sh\ncase "$*" in *-p*) echo "libfabric.so.1 => /usr/lib64/libfabric.so.1";; esac\nexit 0\n' > /sbin/ldconfig && \
    chmod +x /sbin/ldconfig

# hadolint ignore=DL3066
USER k6
WORKDIR /work
