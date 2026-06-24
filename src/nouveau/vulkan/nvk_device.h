/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_DEVICE_H
#define NVK_DEVICE_H 1

#include "nvk_private.h"

#include "nvk_edb_bview_cache.h"
#include "nvk_descriptor_table.h"
#include "nvk_heap.h"
#include "nvk_queue.h"
#include "nvk_upload_queue.h"
#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct nvk_physical_device;
struct nvkmd_dev;
struct nvkmd_mem;
struct vk_pipeline_cache;

enum nvk_dispatch_table {
   NVK_DEVICE_DISPATCH_TABLE,
   NVK_APP_DISPATCH_TABLE,
   NVK_HUD_DISPATCH_TABLE,
   NVK_DISPATCH_TABLE_COUNT,
};

struct nvk_layer_dispatch_tables {
   struct vk_device_dispatch_table app;
   struct vk_device_dispatch_table hud;
};

struct nvk_slm_area {
   simple_mtx_t mutex;
   struct nvkmd_mem *mem;
   uint32_t bytes_per_warp;
   uint32_t bytes_per_tpc;
};

struct nvkmd_mem *
nvk_slm_area_get_mem_ref(struct nvk_slm_area *area,
                         uint32_t *bytes_per_warp_out,
                         uint32_t *bytes_per_mp_out);

struct nvk_object_counts {
   uint32_t fences;
   uint32_t semaphores;
   uint32_t events;
   uint32_t query_pools;
   uint32_t buffers;
   uint32_t buffer_views;
   uint32_t images;
   uint32_t image_views;
   uint32_t samplers;
   uint32_t shader_modules;
   uint32_t pipeline_caches;
   uint32_t graphics_pipelines;
   uint32_t compute_pipelines;
   uint32_t pipeline_layouts;
   uint32_t descriptor_pools;
   uint32_t descriptor_set_layouts;
   uint32_t descriptor_update_templates;
   uint32_t framebuffers;
   uint32_t render_passes;
   uint32_t command_pools;
   uint32_t sampler_ycbcr_conversions;
};

struct nvk_rusd_stats {
   struct timespec time;
   uint32_t temp_gpu;
   uint32_t temp_hbm;
   uint32_t power_gpu;
   uint32_t power_gpu_average;
   uint32_t power_board;
   uint32_t power_board_average;
   uint32_t power_vram_average;
   uint32_t power_cpu;
   uint32_t power_cap;
   uint32_t power_limit_requested;
   uint32_t clock_graphics;
   uint32_t clock_memory;
   uint32_t clock_video;
   uint32_t clock_sm;
   uint32_t util_gpu;
   uint32_t util_memory;
   uint32_t util_nvenc;
   uint32_t util_nvdec;
   uint32_t util_nvjpg;
   uint32_t util_nvofa;
   uint32_t util_nvenc_period;
   uint32_t util_nvdec_period;
   uint32_t util_nvjpg_period;
   uint32_t util_nvofa_period;
   uint32_t pstate;
   uint32_t throttle_status;
   uint32_t throttle_gpu_idle;
   uint32_t throttle_app_clock;
   uint32_t throttle_sw_power_cap;
   uint32_t throttle_hw_slowdown;
   uint32_t throttle_sync_boost;
   uint32_t throttle_sw_thermal;
   uint32_t throttle_hw_thermal;
   uint32_t throttle_hw_power_brake;
   uint32_t throttle_display_clock;
};
struct nvk_device {
   struct vk_device vk;

   struct nvkmd_dev *nvkmd;

   struct nvk_upload_queue upload;

   struct nvk_queue *gfx_queue;
   struct nvk_object_counts obj_counts;
   struct nvk_rusd_stats rusd;
   struct timespec rusd_timestamp;

   struct nvk_layer_dispatch_tables layer_dispatch;
   struct nvkmd_mem *zero_page;
   struct nvk_descriptor_table images;
   struct nvk_descriptor_table samplers;
   struct nvk_edb_bview_cache edb_bview_cache;
   struct nvk_heap shader_heap;
   struct nvk_heap event_heap;
   struct nvk_heap qmd_heap;
   struct nvk_slm_area slm;
   struct nvkmd_mem *vab_memory;

   struct u_printf_ctx printf;

   struct vk_meta_device meta;

   struct nvk_shader *copy_queries;
   struct nvk_shader *copy_indirect;
};

VK_DEFINE_HANDLE_CASTS(nvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

VkResult nvk_device_ensure_slm(struct nvk_device *dev,
                               uint32_t slm_bytes_per_lane,
                               uint32_t crs_bytes_per_warp);

static inline const struct nvk_physical_device *
nvk_device_physical(const struct nvk_device *dev)
{
   return (struct nvk_physical_device *)dev->vk.physical;
}

static inline struct nvk_physical_device *
nvk_device_physical_mut(struct nvk_device *dev)
{
   return (struct nvk_physical_device *)dev->vk.physical;
}

VkResult nvk_device_init_meta(struct nvk_device *dev);
void nvk_device_finish_meta(struct nvk_device *dev);

#endif
