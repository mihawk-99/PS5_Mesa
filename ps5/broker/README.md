# Optional PS5 GPU broker backend

These patches are the ordered, opt-in broker overlay for RADV base revision
`7b59ef27c1b09b9671bc4153c41940c3155c3af2`. PS5_Vulkan's
`tools/build-gpu-broker-client.sh` applies them to isolated source copies and
replaces platform/WSI archive members. Ordinary `-Dradv-winsys=ps5` title
builds retain their native GPU and VideoOut backend.

Apply features, local-memory, client-wait, then wsi. The overlay disables
sparse binding, user pointers and capture/replay; uses a 2 MiB command ring;
exposes broker device-local and shared host-visible memory; and presents
linear images through the owning title. Runtime transport and display code
belong to PS5_Vulkan, with no Wine dependency. Function names prefixed
`ps5_gpu_client_` are that foundation's client ABI.

`client-host-memory-winsys.patch`, applied after `client-identity-winsys.patch`,
gives user pointers back to a client in the owner's own process: the GPU runs in
the owner's address space, so only that client's host memory is reachable
(`ps5_gpu_client_host_memory`; VK_EXT_external_memory_host, which Direct3D 12's
existing heaps need). A client in a process of its own still offers none.

The source of these patches is PS5_Proton commit `878c1e9` (LGPL-2.1-or-later
project changes to MIT Mesa sources). They preserve that attribution and
license. No console address, binary or game data is included.

The original backend passed four DXVK rendering processes (two children and
two descendants), 96 submissions and 27 display flips with correct pixels
and 6/6 native processes absent in `payload-graphics-10`. Extracting it into
this repository is build-qualified; it is not a new console qualification.
The WSI rejection diagnostic and Zink's advertised-mode fix still await
console validation. Native EGL/OpenGL frontend support is not implemented.
