# Validação do pipeline real — 2026-09-16-run-018-frames

Revisão: `b46bd1e9` · Frames: 3 · Pico de VRAM: 5.56 GB (budget: 8.0 GB) · Pico de RSS do host: 4.78 GB

Ver `manifest.json` para configuração completa e log de estágios.

## corridor-02-04234

![corridor-02-04234](frames/corridor-02-04234/regions-overlay.png)

**scene_type:** hallway · **regiões canônicas:** 53 (21 publicadas, 32 contexto estrutural) · **relações:** 509 · **falhas de interpretação:** 0 · **audit:** ✅ pass (38 warnings)

## corridor-02-04595

**FALHOU:** `BackendExecutionError("O backend 'gemini_robotics_er' falhou durante a consulta region: The read operation timed out")`

## corridor-02-04955

**FALHOU:** `BackendExecutionError("O backend 'featup' falhou durante a elevação aprendida FeatUp/JBU: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacity of 7.65 GiB of which 1.14 GiB is free. Process 1581 has 4.34 MiB memory in use. Including non-PyTorch memory, this process has 6.12 GiB memory in use. Of the allocated memory 5.56 GiB is allocated by PyTorch, and 439.50 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)")`
