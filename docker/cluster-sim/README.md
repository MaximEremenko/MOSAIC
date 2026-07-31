# Simulated heterogeneous GPU cluster

Verifies MOSAIC's HPC strategy on one box: three cpuset/cgroup-limited
"nodes" with different GPU/CPU/RAM shapes (1 GB RAM per core), a real
NFSv4 server for the shared filesystem, and netem-shaped interconnect
latency/bandwidth. What SLURM's cgroup plugin does to a job, this does
to a container — including the traps (`os.cpu_count()` and
`/proc/meminfo` describe the host, not the allocation; see
`core/runtime/cpu_resources.py`).

| service | GPUs | cores (cpuset) | RAM | role |
|---|---|---|---|---|
| nfs-server | – | shared | – | exports `$SIM_ROOT/shared` (NFSv4, sync) |
| node1 | 1 (dev 0) | 6 (0–5) | 6 GB | laptop-class |
| node2 | 2 (dev 1,2) | 16 (8–23) | 16 GB | mid node |
| node3 | 1 (dev 3) | 24 (24–47) | 24 GB | fat CPU node |

Interconnect: 250 µs one-way delay + 10 gbit rate on every container
(`SIM_NET_DELAY` / `SIM_NET_RATE` override). Software stack (repo +
venv) is bind-mounted read-only at its host path, like an `/apps` tree.
Node-local scratch is `$SIM_ROOT/scratch/<node>` mounted at `/scratch`.

## Usage

```bash
./up.sh              # build images, modprobe nfsd, compose up (healthchecked)
./prepare-shared.sh  # stage small displacement cases onto the shared FS
sudo docker compose exec node1 bash \
    $PWD/run-case.sh /mnt/shared/configs/displacement/run_parameters_small_all.json
```

`run-case.sh` mirrors `scripts/slurm/submit_hkl40_cases.sbatch`: no
device counts anywhere; each node auto-sizes (one worker per visible
GPU), budgets probe the allocation, durable state and the stage-1
payload store live on `/mnt/shared`.

Strategy checks this cluster has to keep passing:

1. **Laptop floor** — `small_all` end-to-end on node1 (1 GPU, 6 GB).
2. **Shared-store reuse** — second node running the same physics logs
   `Stage-1 payload store already complete` and recomputes nothing.
3. **Elastic restart** — SIGKILL a case mid-residual on node1, rerun on
   node2: checkpoint identity (fixed subchunk slots) holds across
   worker counts; the case completes from durable state.
4. **Array shape** — independent cases on all three nodes concurrently
   against one store (sphere/rod/rest need the `all` case's decoder
   first — same gating as the hkl40 pipeline).

Cleanup: `sudo docker compose down`; shared-FS files created by the
nodes are root-owned on the host (`no_root_squash`), so use `sudo rm`.
