# Documentation

This directory contains release-facing supporting documentation for MOSAIC.

Start here:

- [README.md](../README.md)
  - scientific method overview
  - install story
  - canonical run path
  - release-candidate scope
- [input_schema.md](input_schema.md)
  - canonical release-candidate input schema
  - smoke-scale example structure
  - legacy compatibility mapping
- [hpc_parallel_plan.md](hpc_parallel_plan.md)
  - implemented and deferred pieces of the safer shard/reducer execution model
    for cluster, multi-GPU, pause/resume, and bounded distributed validation
- [cunufft_wrapper_plan.md](cunufft_wrapper_plan.md)
  - staged implementation plan for better cuFINUFFT GPU utilization, batching,
    allocator reuse, wrapper stability, and the bounded benchmark harness
- [examples/README.md](../examples/README.md)
  - supported example files
  - bounded smoke path
  - supported vs research-only example material

Scientific-method language in the main release docs is aligned to the manuscript
sources under `/mnt/c/Projects/MOSAICPaper/Text`, especially the Introduction,
Methods, and Map-Reduce/validation sections.
