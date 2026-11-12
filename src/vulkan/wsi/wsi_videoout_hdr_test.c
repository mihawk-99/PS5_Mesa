/* SPDX-License-Identifier: MIT */
#include <stdio.h>

#include "wsi_videoout_hdr.h"

static int failures;

static void
check(bool ok, const char *what)
{
   if (!ok) {
      fprintf(stderr, "FAIL: %s\n", what);
      failures++;
   }
}

int
main(void)
{
   uint64_t token = 0;
   check(!videoout_hdr_declared(0), "attribute 0 is SDR");
   check(!videoout_hdr_declared(0x40000000), "other attribute bits are SDR");
   check(videoout_hdr_declared(0x20000001), "bit 0x20000000 declares HDR");
   check(!videoout_hdr_pixel_format(false, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                    VK_COLOR_SPACE_HDR10_ST2084_EXT, &token),
         "no PQ token without the title declaration");

   token = 0;
   check(videoout_hdr_pixel_format(true, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                   VK_COLOR_SPACE_HDR10_ST2084_EXT, &token) &&
            token == UINT64_C(0x8100070422000000),
         "A2B10G10R10 maps to the RGB10A2 PQ token");
   token = 0;
   check(videoout_hdr_pixel_format(true, VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                   VK_COLOR_SPACE_HDR10_ST2084_EXT, &token) &&
            token == UINT64_C(0x8100070400000000),
         "A2R10G10B10 maps to the BGR10A2 PQ token");

   check(!videoout_hdr_pixel_format(true, VK_FORMAT_B8G8R8A8_UNORM,
                                    VK_COLOR_SPACE_HDR10_ST2084_EXT, &token),
         "8-bit formats have no PQ token");
   check(!videoout_hdr_pixel_format(true, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, &token),
         "sRGB colour space has no PQ token");

   if (failures)
      return 1;
   printf("wsi_videoout_hdr: all checks passed\n");
   return 0;
}
