#!/bin/bash
set -euo pipefail

rm -f /export/.nfs-ready

# Interconnect shaping: every byte to/from the storage node pays this.
tc qdisc replace dev eth0 root netem \
    delay "${SIM_NET_DELAY:-250us}" rate "${SIM_NET_RATE:-10gbit}" limit 10000 \
    || echo "WARN: tc netem failed (no NET_ADMIN?)" >&2

rpcbind || true
mount -t nfsd nfsd /proc/fs/nfsd

echo "/export *(rw,sync,no_subtree_check,no_root_squash,fsid=0)" > /etc/exports
exportfs -r
rpc.nfsd --no-nfs-version 3 8
rpc.mountd --no-nfs-version 3 --foreground &

touch /export/.nfs-ready
echo "READY nfs-server: exporting /export (NFSv4, sync)"
wait
