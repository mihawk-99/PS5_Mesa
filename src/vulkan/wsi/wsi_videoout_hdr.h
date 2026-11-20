/* SPDX-License-Identifier: MIT */
#ifndef WSI_VIDEOOUT_HDR_H
#define WSI_VIDEOOUT_HDR_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#define VIDEOOUT_ATTRIBUTE_HDR 0x20000000u
/* These PQ buffer tokens were observed in forever, not defined by a public API. */
#define VIDEOOUT_RGB10A2_PQ UINT64_C(0x8100070422000000)
#define VIDEOOUT_BGR10A2_PQ UINT64_C(0x8100070400000000)

static inline bool
videoout_hdr_declared(uint32_t attribute)
{
   return (attribute & VIDEOOUT_ATTRIBUTE_HDR) != 0;
}

static inline bool
videoout_hdr_pixel_format(bool declared, VkFormat format, VkColorSpaceKHR color_space, uint64_t *token)
{
   if (!declared || color_space != VK_COLOR_SPACE_HDR10_ST2084_EXT)
      return false;
   if (format == VK_FORMAT_A2B10G10R10_UNORM_PACK32)
      *token = VIDEOOUT_RGB10A2_PQ;
   else if (format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)
      *token = VIDEOOUT_BGR10A2_PQ;
   else
      return false;
   return true;
}

#endif
