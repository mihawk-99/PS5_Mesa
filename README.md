# RADV on the PlayStation 5

This is my fork of Mesa 26.2.0; all of its work is on branch `main`. It
runs RADV, Mesa's Vulkan driver for AMD GPUs, with its ACO shader compiler, on
the PlayStation 5's GPU from a homebrew title. The fork adds a *winsys* for the
console, the layer RADV keeps the kernel behind, and changes RADV only where
the console differs from a Linux GPU. Mesa's own README is
[README.rst](README.rst).

It is route B of
[PS5_Vulkan](https://github.com/mihawk-99/PS5_Vulkan)'s
`docs/VULKAN_1_4_PLAN.md`. The goal is a genuine Vulkan 1.4 driver that passes
the Khronos CTS on the console. It reports Vulkan 1.4, and the pinned 1.4 CTS
(vulkan-cts-1.4.6.2) runs on the console in full; what it does not pass yet is
listed, with its reason, in PS5_Vulkan's `docs/CTS_GAPS.md`.

It is what the titles ship: vkQuake since 2026-09-28, and
[PS5 RetroArch](https://github.com/mihawk-99/PS5_RetroArch) since
v0.5.0-alpha.5, with its PSP, GameCube/Wii and PS2 cores rendering through it.

## The winsys (`src/amd/vulkan/winsys/ps5`)

| Part | What it does |
| --- | --- |
| `radv_ps5_platform.c` | GPU-visible direct memory. The CPU and the GPU see each allocation at one address. 32-bit buffers (shaders, descriptors) go in a 4 GiB window at 0x200000000, whose top 256 MiB is kept for ray-tracing capture and replay; other buffers go in a 256 GiB region at 0x4000000000. It also submits through `sceAgcDriverSubmitDcb` with the suspend point, and flushes the CPU's caches. A host build models all of it, so RADV runs on a PC for everything that does not need the GPU. |
| `radv_ps5_winsys.c` | The GPU description, in the amdgpu kernel's own form, completed by `ac_gpu_info`'s derivations: GFX10.3 shader cores beside GC 10.1.3's fixed-function blocks, addressed as the console's tile maps were measured. It holds the process-wide queue, which the completion markers track. |
| `radv_ps5_cs.c` | Command streams kept in CPU memory. A submission copies all of them into one AGC submission, because the console faults a PM4 `INDIRECT_BUFFER` into title memory, and ends with a cache-flushing `RELEASE_MEM` that writes its sequence number. |
| `radv_ps5_bo.c` | Buffers: one direct-memory allocation each, always resident. |
| `radv_ps5_sync.c` | A binary sync object signalled by a submission's sequence number. Mesa's runtime builds timeline semaphores over it. |

The swapchain is `src/vulkan/wsi/wsi_common_videoout.c`: `VK_KHR_display` on
the console's VideoOut. It opens the output once per process, flips at vblank,
and switches to 119.88 Hz where the title declares it and the display really
refreshes at it (it measures, and goes back to 59.94 Hz otherwise). A
swapchain may be smaller than the 3840x2160 mode: each size gets its own set
of VideoOut framebuffers, which VideoOut scales to the whole output. A
1920x1080 swapchain was measured filling the screen (DXVK D3D11 and D3D9
through Wine, and a 1080p to 4K resize in one process); other sizes are
accepted but not measured yet.

## Where RADV changes

Each change is general Vulkan behaviour for any application, not a case for one
title:

- **Geometry and tessellation shaders run as compute** where the console's
  fixed-function blocks cannot run them, feeding the rasteriser from buffers.
- **Mesh and task shaders.** The GPU has neither per-primitive NGG parameters
  nor the command processor's task and mesh dispatch packets: mesh workgroups
  run in parts that share one run's outputs through a publish ring, and task
  shaders run on the graphics ring in chunks.
- **Queues.** Compute and transfer queue families are served from the graphics
  ring.
- **Memory types** for an integrated GPU: host-cached types and host pointers.
- **Ray tracing**: pipelines, ray queries, and group handles captured and
  replayed at their captured addresses.
- **The shader cache** (`src/util`): Mesa's disk cache in its database form, in
  the title's folder (`/app0/radv-shader-cache`). On the PS5 its folders and
  files stay open to the console's FTP service (0777 and 0666), a lookup does
  not create cache parts, and the database is opened exclusively, once per
  process: the console's file operations are slow, and titles read the cache
  from many threads at once (`MESA_DISK_CACHE_DATABASE_EXCLUSIVE`).

The branches these came in on (`ps5-gs-compute`, `ps5-gs-tess`, `ps5-mesh`,
`ps5-task`, `ps5-queues`, `ps5-memtypes`, `ps5-wsi`, `ps5-rt-replay`,
`ps5-shader-cache` and others) are merged into `main`, and PS5_Vulkan's
`docs/RADV_PHASE.md` is the round-by-round record, with what the console
measured for each.

What the winsys and these changes rely on was measured on the console, in
PS5_Vulkan's probes and CTS runs. Gaps in the console's platform (kernel
declarations, libc functions, the title heap) are not patched here: they live
in the shared platform layer of my payload SDK fork,
[PS5_PayloadSDK](https://github.com/mihawk-99/PS5_PayloadSDK).

## Building

`meson setup` with `-Dradv-winsys=ps5`, for the console with
`PS5_Vulkan/tooling/radv/ps5-cross.ini` and a static default library.
`PS5_Vulkan/tools/build-radv.sh` builds the revision `PS5_Vulkan` pins (a debug
archive with Mesa's assertions for the smoke test and the CTS, and
`build-radv.sh release` for the titles) and records it, and
`PS5_Vulkan/tools/radv-link.sh` links the archive into a title. The build takes
zlib from Mesa's subproject, for the shader cache.

A Linux host build of the same option (a shared ICD for the Khronos loader)
models the console. It runs API-level tests, the CTS's API groups and pipeline
compiles on a PC.

## Licence

Mesa's licences apply. The files this fork adds are MIT, like the code around
them.
