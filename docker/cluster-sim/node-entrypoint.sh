#!/bin/bash
set -euo pipefail

# Interconnect shaping (egress; symmetric because every peer shapes too).
tc qdisc replace dev eth0 root netem \
    delay "${SIM_NET_DELAY:-250us}" rate "${SIM_NET_RATE:-10gbit}" limit 10000 \
    || echo "WARN: tc netem failed (no NET_ADMIN?)" >&2

mkdir -p /mnt/shared /scratch
for _ in $(seq 1 90); do
    if mount -t nfs4 -o vers=4.2,proto=tcp,hard,timeo=600,lookupcache=positive \
        nfs-server:/ /mnt/shared 2>/tmp/mount.err; then
        break
    fi
    sleep 2
done
if ! mountpoint -q /mnt/shared; then
    echo "FATAL: NFS mount failed:" >&2
    cat /tmp/mount.err >&2
    exit 1
fi

/usr/sbin/sshd -e 2>/dev/null || echo "WARN: sshd failed to start" >&2

echo "READY node=$(hostname) cpus=$(nproc) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)" \
     "cgroup_mem=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '?')"
exec sleep infinity
