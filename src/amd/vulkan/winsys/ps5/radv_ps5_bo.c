/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_winsys.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/os_time.h"
#include "util/u_atomic.h"

/* A buffer is one direct-memory allocation, mapped for the CPU and the GPU at
 * the same address for as long as it lives, so it is always resident, its GPU
 * address is its CPU address, and mapping it is a view of it (PS5_Vulkan B3). A
 * 32-bit buffer lies in the shaders' address window; any other anywhere below
 * 2^47 (R86-R88). */

uint64_t radv_ps5_allocated_bytes;

/* A buffer freed while the GPU may still run submissions made before the free
 * keeps its memory and its range until they have run, as the amdgpu kernel
 * keeps a freed buffer's pages until its fences signal. Unmapped at once, a
 * read the GPU had still to make faults it (the title is ended), and the range
 * may meanwhile go to another buffer. */
struct radv_ps5_retired {
   struct radv_ps5_memory memory;
   uint64_t seq;
   bool sparse;
};
static simple_mtx_t radv_ps5_retired_lock = SIMPLE_MTX_INITIALIZER;
static struct util_dynarray radv_ps5_retired;

static void
radv_ps5_memory_release(struct radv_ps5_memory *memory, bool sparse)
{
   if (sparse)
      radv_ps5_vrange_release(memory);
   else
      radv_ps5_memory_free(memory);
}

/* Gives back what the GPU has finished with; the newest submission the rest
 * waits for, or 0 if nothing is held. */
static uint64_t
radv_ps5_retired_free(struct radv_ps5_queue *queue)
{
   simple_mtx_lock(&radv_ps5_retired_lock);
   const uint64_t completed = radv_ps5_queue_poll(queue);
   uint64_t newest = 0;
   unsigned kept = 0;
   util_dynarray_foreach (&radv_ps5_retired, struct radv_ps5_retired, entry) {
      if (entry->seq <= completed) {
         radv_ps5_memory_release(&entry->memory, entry->sparse);
      } else {
         newest = MAX2(newest, entry->seq);
         *util_dynarray_element(&radv_ps5_retired, struct radv_ps5_retired, kept++) = *entry;
      }
   }
   radv_ps5_retired.size = kept * sizeof(struct radv_ps5_retired);
   simple_mtx_unlock(&radv_ps5_retired_lock);
   return newest;
}

static void
radv_ps5_memory_retire(struct radv_ps5_queue *queue, struct radv_ps5_memory *memory, bool sparse)
{
   const uint64_t seq = p_atomic_read(&queue->submitted_seq);
   if (seq <= radv_ps5_queue_poll(queue)) {
      radv_ps5_memory_release(memory, sparse);
   } else {
      simple_mtx_lock(&radv_ps5_retired_lock);
      const struct radv_ps5_retired entry = {.memory = *memory, .seq = seq, .sparse = sparse};
      util_dynarray_append(&radv_ps5_retired, entry);
      simple_mtx_unlock(&radv_ps5_retired_lock);
   }
   radv_ps5_retired_free(queue);
}

/* After an allocation failed: waits for the GPU to finish with the memory held
 * for it and gives it back; false if nothing was held. */
static bool
radv_ps5_retired_wait(struct radv_ps5_queue *queue)
{
   const uint64_t newest = radv_ps5_retired_free(queue);
   if (!newest || !radv_ps5_queue_wait_seq(queue, newest, OS_TIMEOUT_INFINITE))
      return false;
   radv_ps5_retired_free(queue);
   return true;
}

/* Memory reports (VK_EXT_device_memory_report) name a buffer by an id that is
 * never 0 and never reused. */
static uint64_t radv_ps5_next_obj_id;

static VkResult
radv_ps5_buffer_create(struct radeon_winsys *rws, uint64_t size, unsigned alignment, enum radeon_bo_domain domain,
                       enum radeon_bo_flag flags, unsigned priority, uint64_t replay_address,
                       struct radeon_winsys_bo **out_bo)
{
   (void)priority;
   struct radv_ps5_winsys *const ws = radv_ps5_winsys(rws);
   *out_bo = NULL;

   /* A sparse resource: a range with nothing bound yet (radv_ps5_vrange_*),
    * placed for capture and replay as any other buffer is. */
   if (flags & RADEON_FLAG_VIRTUAL) {
      struct radv_ps5_bo *const bo = calloc(1, sizeof(*bo));
      if (!bo)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      bool reserved;
      do {
         reserved = radv_ps5_vrange_reserve(size, flags & RADEON_FLAG_32BIT,
                                            (flags & RADEON_FLAG_REPLAYABLE) || replay_address, replay_address,
                                            &bo->memory);
      } while (!reserved && radv_ps5_retired_wait(ws->queue));
      if (!reserved) {
         free(bo);
         return replay_address ? VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
      bo->ws = ws;
      bo->flags = flags;
      bo->sparse = true;
      bo->base.va = (uint64_t)(uintptr_t)bo->memory.cpu;
      bo->base.size = size;
      bo->base.is_local = true;
      bo->base.use_global_list = true;
      bo->base.initial_domain = domain;
      bo->base.obj_id = p_atomic_inc_return(&radv_ps5_next_obj_id);
      *out_bo = &bo->base;
      return VK_SUCCESS;
   }
   /* Capture and replay: a captured buffer outside the window goes at the top
    * of the device-memory region, and a replayed one at its captured address
    * or nowhere (radv_ps5_memory_alloc_replayable). */
   const bool window32 = flags & RADEON_FLAG_32BIT;
   const bool replayable = (flags & RADEON_FLAG_REPLAYABLE) || replay_address;

   struct radv_ps5_bo *const bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   bool allocated;
   do {
      allocated = replayable ? radv_ps5_memory_alloc_replayable(size, alignment, window32, replay_address, &bo->memory)
                             : radv_ps5_memory_alloc(size, alignment, window32, &bo->memory);
   } while (!allocated && radv_ps5_retired_wait(ws->queue));
   if (!allocated) {
      free(bo);
      return replay_address ? VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   bo->ws = ws;
   bo->flags = flags;
   bo->base.va = (uint64_t)(uintptr_t)bo->memory.cpu;
   bo->base.size = size;
   bo->base.is_local = true;
   bo->base.use_global_list = true;
   bo->base.initial_domain = domain;
   bo->base.obj_id = p_atomic_inc_return(&radv_ps5_next_obj_id);

   /* Direct memory arrives with whatever the pool held. amdgpu gives GTT
    * buffers zeroed pages and VRAM ones zeroes when asked, and RADV relies on
    * both: an event's buffer is never written before its first status read
    * (dEQP-VK.api.command_buffers.submit_count_equal_zero read a stale 1).
    * The GPU reads through the CPU's writes only once they are in memory. */
   if ((flags & RADEON_FLAG_ZERO_VRAM) || (domain & RADEON_DOMAIN_GTT)) {
      memset(bo->memory.cpu, 0, size);
      radv_ps5_cpu_flush(bo->memory.cpu, size);
   }

   p_atomic_add(&radv_ps5_allocated_bytes, bo->memory.bytes);
   if (domain & RADEON_DOMAIN_VRAM)
      p_atomic_add(&ws->allocated_vram, bo->memory.bytes);
   else
      p_atomic_add(&ws->allocated_gtt, bo->memory.bytes);

   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static void
radv_ps5_buffer_destroy(struct radeon_winsys *rws, struct radeon_winsys_bo *base)
{
   struct radv_ps5_winsys *const ws = radv_ps5_winsys(rws);
   struct radv_ps5_bo *const bo = radv_ps5_bo(base);
   if (!bo)
      return;
   if (bo->sparse) {
      radv_ps5_memory_retire(ws->queue, &bo->memory, true);
      free(bo);
      return;
   }
   /* The application's memory stays its own. */
   if (bo->imported) {
      free(bo);
      return;
   }
   p_atomic_add(&radv_ps5_allocated_bytes, -(int64_t)bo->memory.bytes);
   if (bo->base.initial_domain & RADEON_DOMAIN_VRAM)
      p_atomic_add(&ws->allocated_vram, -(int64_t)bo->memory.bytes);
   else
      p_atomic_add(&ws->allocated_gtt, -(int64_t)bo->memory.bytes);
   if (bo->placed)
      radv_ps5_memory_unmap_at(bo->placed, bo->memory.bytes, false);
   radv_ps5_memory_retire(ws->queue, &bo->memory, false);
   free(bo);
}

static void *
radv_ps5_buffer_map(struct radeon_winsys *rws, struct radeon_winsys_bo *base, bool use_fixed_addr, void *fixed_addr)
{
   (void)rws;
   struct radv_ps5_bo *const bo = radv_ps5_bo(base);
   if (bo->sparse)
      return NULL;
   if (!use_fixed_addr)
      return bo->memory.cpu;

   /* A placed mapping (VK_EXT_map_memory_placed): the same direct memory mapped
    * again for the CPU, at the address the application chose. The GPU keeps
    * the first mapping's address. */
   if (bo->placed || !radv_ps5_memory_map_at(&bo->memory, fixed_addr))
      return NULL;
   bo->placed = fixed_addr;
   return fixed_addr;
}

static void
radv_ps5_buffer_unmap(struct radeon_winsys *rws, struct radeon_winsys_bo *base, bool replace)
{
   (void)rws;
   struct radv_ps5_bo *const bo = radv_ps5_bo(base);
   /* The first mapping lasts as long as the buffer; only a placed one goes. */
   if (!bo->placed)
      return;
   radv_ps5_memory_unmap_at(bo->placed, bo->memory.bytes, replace);
   bo->placed = NULL;
}

static VkResult
radv_ps5_buffer_from_ptr(struct radeon_winsys *rws, void *pointer, uint64_t size, unsigned priority,
                         struct radeon_winsys_bo **out_bo)
{
   (void)priority;
   struct radv_ps5_winsys *const ws = radv_ps5_winsys(rws);
   *out_bo = NULL;

   /* Host memory the application owns, at the address the CPU uses, once it
    * has GPU access. */
   const uint64_t va = (uint64_t)(uintptr_t)pointer;
   if (!size || va % RADV_PS5_PAGE_BYTES || size % RADV_PS5_PAGE_BYTES || va >= RADV_PS5_GPU_ADDRESS_LIMIT ||
       size > RADV_PS5_GPU_ADDRESS_LIMIT - va || !radv_ps5_memory_grant_gpu(pointer, size))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   struct radv_ps5_bo *const bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   bo->ws = ws;
   bo->imported = true;
   bo->memory = (struct radv_ps5_memory){.cpu = pointer, .bytes = size, .physical = -1};
   bo->base.va = va;
   bo->base.size = size;
   bo->base.is_local = false;
   bo->base.use_global_list = true;
   bo->base.initial_domain = RADEON_DOMAIN_GTT;
   bo->base.obj_id = p_atomic_inc_return(&radv_ps5_next_obj_id);
   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static VkResult
radv_ps5_buffer_from_fd(struct radeon_winsys *rws, int fd, unsigned priority, struct radeon_winsys_bo **out_bo,
                        uint64_t *alloc_size)
{
   (void)rws;
   (void)fd;
   (void)priority;
   (void)alloc_size;
   *out_bo = NULL;
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static bool
radv_ps5_buffer_get_fd(struct radeon_winsys *rws, struct radeon_winsys_bo *bo, int *fd)
{
   (void)rws;
   (void)bo;
   (void)fd;
   return false;
}

static bool
radv_ps5_buffer_get_flags_from_fd(struct radeon_winsys *rws, int fd, enum radeon_bo_domain *domains,
                                  enum radeon_bo_flag *flags)
{
   (void)rws;
   (void)fd;
   (void)domains;
   (void)flags;
   return false;
}

static void
radv_ps5_buffer_set_metadata(struct radeon_winsys *rws, struct radeon_winsys_bo *base, struct radeon_bo_metadata *md)
{
   (void)rws;
   radv_ps5_bo(base)->metadata = *md;
}

static void
radv_ps5_buffer_get_metadata(struct radeon_winsys *rws, struct radeon_winsys_bo *base, struct radeon_bo_metadata *md)
{
   (void)rws;
   *md = radv_ps5_bo(base)->metadata;
}

static VkResult
radv_ps5_buffer_virtual_bind(struct radeon_winsys *rws, struct radeon_winsys_bo *parent, uint64_t offset,
                             uint64_t size, struct radeon_winsys_bo *bo, uint64_t bo_offset)
{
   (void)rws;
   struct radv_ps5_bo *const range = radv_ps5_bo(parent);
   if (!range->sparse || offset > range->memory.bytes || size > range->memory.bytes - offset)
      return VK_ERROR_UNKNOWN;
   uint8_t *const at = (uint8_t *)range->memory.cpu + offset;
   const bool done = bo ? radv_ps5_vrange_bind(at, size, &radv_ps5_bo(bo)->memory, bo_offset)
                        : radv_ps5_vrange_unbind(at, size);
   if (!done)
      fprintf(stderr, "radv/ps5: sparse %s of %" PRIu64 " bytes at %p failed\n", bo ? "bind" : "unbind", size,
              (void *)at);
   return done ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

static VkResult
radv_ps5_buffer_make_resident(struct radeon_winsys *rws, struct radeon_winsys_bo *bo, bool resident)
{
   (void)rws;
   (void)bo;
   (void)resident;
   return VK_SUCCESS;
}

static bool
radv_ps5_bo_wait_for_idle(struct radeon_winsys *rws, struct radeon_winsys_bo *bo)
{
   (void)bo;
   struct radv_ps5_queue *const queue = radv_ps5_winsys(rws)->queue;
   return radv_ps5_queue_wait_seq(queue, p_atomic_read(&queue->submitted_seq), OS_TIMEOUT_INFINITE);
}

static void
radv_ps5_dump_bo_ranges(struct radeon_winsys *rws, FILE *file)
{
   (void)rws;
   (void)file;
}

static void
radv_ps5_dump_bo_log(struct radeon_winsys *rws, FILE *file)
{
   (void)rws;
   (void)file;
}

void
radv_ps5_bo_init_functions(struct radv_ps5_winsys *ws)
{
   ws->base.buffer_create = radv_ps5_buffer_create;
   ws->base.buffer_destroy = radv_ps5_buffer_destroy;
   ws->base.buffer_map = radv_ps5_buffer_map;
   ws->base.buffer_unmap = radv_ps5_buffer_unmap;
   ws->base.buffer_from_ptr = radv_ps5_buffer_from_ptr;
   ws->base.buffer_from_fd = radv_ps5_buffer_from_fd;
   ws->base.buffer_get_fd = radv_ps5_buffer_get_fd;
   ws->base.buffer_get_flags_from_fd = radv_ps5_buffer_get_flags_from_fd;
   ws->base.buffer_set_metadata = radv_ps5_buffer_set_metadata;
   ws->base.buffer_get_metadata = radv_ps5_buffer_get_metadata;
   ws->base.buffer_virtual_bind = radv_ps5_buffer_virtual_bind;
   ws->base.buffer_make_resident = radv_ps5_buffer_make_resident;
   ws->base.bo_wait_for_idle = radv_ps5_bo_wait_for_idle;
   ws->base.dump_bo_ranges = radv_ps5_dump_bo_ranges;
   ws->base.dump_bo_log = radv_ps5_dump_bo_log;
}
