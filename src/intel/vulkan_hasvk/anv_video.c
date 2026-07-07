/*
 * Copyright © 2021 Red Hat
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "anv_private.h"

#include <stdlib.h>
#include <string.h>
#include "vk_common_entrypoints.h"
#include "vk_video/vulkan_video_codecs_common.h"
#include "vk_video/vulkan_video_codec_h264std.h"

VkResult
anv_CreateVideoSessionKHR(VkDevice _device,
                          const VkVideoSessionCreateInfoKHR *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkVideoSessionKHR *pVideoSession)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   struct anv_video_session *vid =
      vk_alloc2(&device->vk.alloc, pAllocator, sizeof(*vid), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!vid)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   memset(vid, 0, sizeof(struct anv_video_session));

   VkResult result = vk_video_session_init(&device->vk,
                                           &vid->vk,
                                           pCreateInfo);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, vid);
      return result;
   }

   *pVideoSession = anv_video_session_to_handle(vid);
   return VK_SUCCESS;
}

void
anv_DestroyVideoSessionKHR(VkDevice _device,
                           VkVideoSessionKHR _session,
                           const VkAllocationCallbacks *pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_video_session, vid, _session);
   if (!_session)
      return;

   /* Otherwise we hit VK_ERROR_DEVICE_LOST for unknown reasons. */
   vk_common_DeviceWaitIdle(_device);

   vk_video_session_finish(&vid->vk);
   vk_free2(&device->vk.alloc, pAllocator, vid);
}

VkResult
anv_CreateVideoSessionParametersKHR(VkDevice _device,
                                    const
                                    VkVideoSessionParametersCreateInfoKHR
                                    *pCreateInfo,
                                    const VkAllocationCallbacks *pAllocator,
                                    VkVideoSessionParametersKHR
                                    *pVideoSessionParameters)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   struct vk_video_session_parameters *params =
      vk_video_session_parameters_create(&device->vk, pCreateInfo, pAllocator,
                                         sizeof(struct
                                                anv_video_session_params));
   if (!params)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pVideoSessionParameters = vk_video_session_parameters_to_handle(params);
   return VK_SUCCESS;
}

void
anv_DestroyVideoSessionParametersKHR(VkDevice _device,
                                     VkVideoSessionParametersKHR _params,
                                     const VkAllocationCallbacks *pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   VK_FROM_HANDLE(vk_video_session_parameters, params, _params);

   vk_video_session_parameters_destroy(&device->vk, pAllocator, params);
}

VkResult
anv_GetPhysicalDeviceVideoCapabilitiesKHR(VkPhysicalDevice physicalDevice,
                                          const VkVideoProfileInfoKHR
                                          *pVideoProfile,
                                          VkVideoCapabilitiesKHR
                                          *pCapabilities)
{
   if (pVideoProfile->videoCodecOperation != VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR) {
      return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;
   }

   const VkVideoDecodeH264ProfileInfoKHR *h264_profile =
      vk_find_struct_const(pVideoProfile->pNext, VIDEO_DECODE_H264_PROFILE_INFO_KHR);
   if (h264_profile) {
      switch (h264_profile->stdProfileIdc) {
      case STD_VIDEO_H264_PROFILE_IDC_BASELINE:
      case STD_VIDEO_H264_PROFILE_IDC_MAIN:
      case STD_VIDEO_H264_PROFILE_IDC_HIGH:
         break;
      default:
         return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;
      }
   }

   pCapabilities->minBitstreamBufferOffsetAlignment = 32;
   pCapabilities->minBitstreamBufferSizeAlignment = 32;
   pCapabilities->pictureAccessGranularity.width = ANV_MB_WIDTH;
   pCapabilities->pictureAccessGranularity.height = ANV_MB_HEIGHT;
   pCapabilities->minCodedExtent.width = ANV_MB_WIDTH;
   pCapabilities->minCodedExtent.height = ANV_MB_HEIGHT;
   pCapabilities->maxCodedExtent.width = 4096;
   pCapabilities->maxCodedExtent.height = 4096;
   pCapabilities->flags =
      VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;

   struct VkVideoDecodeCapabilitiesKHR *dec_caps =
      (struct VkVideoDecodeCapabilitiesKHR *)
      vk_find_struct(pCapabilities->pNext, VIDEO_DECODE_CAPABILITIES_KHR);
   if (dec_caps)
      dec_caps->flags =
         VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR;

   struct VkVideoDecodeH264CapabilitiesKHR *ext =
      (struct VkVideoDecodeH264CapabilitiesKHR *)
      vk_find_struct(pCapabilities->pNext,
                     VIDEO_DECODE_H264_CAPABILITIES_KHR);
   pCapabilities->maxDpbSlots = 16;
   pCapabilities->maxActiveReferencePictures = 16;

   if (ext) {
      ext->fieldOffsetGranularity.x = 0;
      ext->fieldOffsetGranularity.y = 0;
      ext->maxLevelIdc = 51;
   }
   strcpy(pCapabilities->stdHeaderVersion.extensionName,
          VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME);
   pCapabilities->stdHeaderVersion.specVersion =
      VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION;

   return VK_SUCCESS;
}

VkResult
anv_GetPhysicalDeviceVideoFormatPropertiesKHR(VkPhysicalDevice physicalDevice,
                                              const
                                              VkPhysicalDeviceVideoFormatInfoKHR
                                              *pVideoFormatInfo,
                                              uint32_t
                                              *pVideoFormatPropertyCount,
                                              VkVideoFormatPropertiesKHR
                                              *pVideoFormatProperties)
{
   *pVideoFormatPropertyCount = 1;

   if (!pVideoFormatProperties)
      return VK_SUCCESS;

   VkImageUsageFlags usage_flags = pVideoFormatInfo->imageUsage;

   const VkImageUsageFlags dpb_output_coincide_usage =
      VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
      VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
   if ((usage_flags & dpb_output_coincide_usage) == VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR ||
       (usage_flags & dpb_output_coincide_usage) == VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
      usage_flags |= dpb_output_coincide_usage;

   if (usage_flags & VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
      usage_flags |=
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

   pVideoFormatProperties[0].sType =
      VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
   pVideoFormatProperties[0].pNext = NULL;
   pVideoFormatProperties[0].format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
   pVideoFormatProperties[0].componentMapping.r =
      VK_COMPONENT_SWIZZLE_IDENTITY;
   pVideoFormatProperties[0].componentMapping.g =
      VK_COMPONENT_SWIZZLE_IDENTITY;
   pVideoFormatProperties[0].componentMapping.b =
      VK_COMPONENT_SWIZZLE_IDENTITY;
   pVideoFormatProperties[0].componentMapping.a =
      VK_COMPONENT_SWIZZLE_IDENTITY;
   pVideoFormatProperties[0].imageCreateFlags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT
      | VK_IMAGE_CREATE_ALIAS_BIT;
   pVideoFormatProperties[0].imageType = VK_IMAGE_TYPE_2D;
   pVideoFormatProperties[0].imageTiling = VK_IMAGE_TILING_OPTIMAL;
   pVideoFormatProperties[0].imageUsageFlags = usage_flags;

   return VK_SUCCESS;
}

static void
get_h264_video_session_mem_reqs(struct anv_video_session *vid,
                                VkVideoSessionMemoryRequirementsKHR *mem_reqs,
                                uint32_t *pVideoSessionMemoryRequirementsCount,
                                uint32_t memory_types)
{
   VK_OUTARRAY_MAKE_TYPED(VkVideoSessionMemoryRequirementsKHR,
                          out,
                          mem_reqs,
                          pVideoSessionMemoryRequirementsCount);

   uint32_t width_in_mb =
      align(vid->vk.max_coded.width, ANV_MB_WIDTH) / ANV_MB_WIDTH;

   /* intra row store is width in macroblocks * 64 */
   vk_outarray_append_typed(VkVideoSessionMemoryRequirementsKHR, &out, p) {
      p->memoryBindIndex = ANV_VID_MEM_H264_INTRA_ROW_STORE;
      p->memoryRequirements.size = width_in_mb * 64;
      p->memoryRequirements.alignment = 64;
      p->memoryRequirements.memoryTypeBits = memory_types;
   }

   /* deblocking filter row store is width in macroblocks * 64 * 4 */
   vk_outarray_append_typed(VkVideoSessionMemoryRequirementsKHR, &out, p) {
      p->memoryBindIndex = ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE;
      p->memoryRequirements.size = width_in_mb * 64 * 4;
      p->memoryRequirements.alignment = 64;
      p->memoryRequirements.memoryTypeBits = memory_types;
   }

   /* bsd mpc row scratch is width in macroblocks * 64 * 2 */
   vk_outarray_append_typed(VkVideoSessionMemoryRequirementsKHR, &out, p) {
      p->memoryBindIndex = ANV_VID_MEM_H264_BSD_MPC_ROW_SCRATCH;
      p->memoryRequirements.size = width_in_mb * 64 * 2;
      p->memoryRequirements.alignment = 64;
      p->memoryRequirements.memoryTypeBits = memory_types;
   }

   /* mpr row scratch is width in macroblocks * 64 * 2 */
   vk_outarray_append_typed(VkVideoSessionMemoryRequirementsKHR, &out, p) {
      p->memoryBindIndex = ANV_VID_MEM_H264_MPR_ROW_SCRATCH;
      p->memoryRequirements.size = width_in_mb * 64 * 2;
      p->memoryRequirements.alignment = 64;
      p->memoryRequirements.memoryTypeBits = memory_types;
   }
}

VkResult
anv_GetVideoSessionMemoryRequirementsKHR(VkDevice _device,
                                         VkVideoSessionKHR videoSession,
                                         uint32_t
                                         *pVideoSessionMemoryRequirementsCount,
                                         VkVideoSessionMemoryRequirementsKHR
                                         *mem_reqs)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_video_session, vid, videoSession);

   uint32_t memory_types = (1ull << device->physical->memory.type_count) - 1;
   switch (vid->vk.op) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
      get_h264_video_session_mem_reqs(vid, mem_reqs,
                                      pVideoSessionMemoryRequirementsCount,
                                      memory_types);
      break;
   default:
      UNREACHABLE("unknown codec");
   }

   return VK_SUCCESS;
}

VkResult
anv_UpdateVideoSessionParametersKHR(VkDevice _device,
                                    VkVideoSessionParametersKHR _params,
                                    const
                                    VkVideoSessionParametersUpdateInfoKHR
                                    *pUpdateInfo)
{
   ANV_FROM_HANDLE(anv_video_session_params, params, _params);
   return vk_video_session_parameters_update(&params->vk, pUpdateInfo);
}

static void
copy_bind(struct anv_vid_mem *dst, const VkBindVideoSessionMemoryInfoKHR *src)
{
   dst->mem = anv_device_memory_from_handle(src->memory);
   dst->offset = src->memoryOffset;
   dst->size = src->memorySize;
}

VkResult
anv_BindVideoSessionMemoryKHR(VkDevice _device,
                              VkVideoSessionKHR videoSession,
                              uint32_t bind_mem_count,
                              const VkBindVideoSessionMemoryInfoKHR *bind_mem)
{
   ANV_FROM_HANDLE(anv_video_session, vid, videoSession);

   switch (vid->vk.op) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
      /* Validate bind_mem_count and memoryBindIndex values to prevent
       * out-of-bounds access to vid->vid_mem array. */
      for (unsigned i = 0; i < bind_mem_count; i++) {
         if (bind_mem[i].memoryBindIndex >= ANV_VID_MEM_H264_MAX) {
            continue;
         }
         copy_bind(&vid->vid_mem[bind_mem[i].memoryBindIndex], &bind_mem[i]);
      }
      break;
   default:
      UNREACHABLE("unknown codec!");
   }
   return VK_SUCCESS;
}
