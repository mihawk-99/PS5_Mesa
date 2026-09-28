/*
 * Copyright © 2026 dancingmirrors@icloud.com
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef ANV_VIDEO_H264_SLICE_H
#define ANV_VIDEO_H264_SLICE_H

#include "vulkan/vulkan.h"
#include "vulkan/util/vk_util.h"

#include "vk_video/vulkan_video_codec_h264std.h"
#include "vk_video/vulkan_video_codec_h264std_decode.h"

#define ANV_H264_MAX_REF_FRAMES 16
#define ANV_H264_SLICE_P 0
#define ANV_H264_SLICE_B 1
#define ANV_H264_SLICE_I 2
#define ANV_H264_SLICE_SP 3
#define ANV_H264_SLICE_SI 4
#define ANV_H264_INVALID_SLOT 0xFF

struct anv_h264_slice_params {
   int first_mb_in_slice;
   int slice_type;
   int num_ref_idx_l0_active_minus1;
   int num_ref_idx_l1_active_minus1;
   int direct_spatial_mv_pred_flag;
   int cabac_init_idc;
   int slice_qp_delta;
   int disable_deblocking_filter_idc;
   int slice_alpha_c0_offset_div2;
   int slice_beta_offset_div2;
   int luma_log2_weight_denom;
   int chroma_log2_weight_denom;

   uint8_t ref_list0[ANV_H264_MAX_REF_FRAMES];
   uint8_t ref_list1[ANV_H264_MAX_REF_FRAMES];

   uint8_t ref_list0_bottom[ANV_H264_MAX_REF_FRAMES];
   uint8_t ref_list1_bottom[ANV_H264_MAX_REF_FRAMES];

   bool has_weight_offsets_l0;
   bool has_weight_offsets_l1;

   int16_t luma_weight_l0[ANV_H264_MAX_REF_FRAMES];
   int16_t luma_offset_l0[ANV_H264_MAX_REF_FRAMES];
   int16_t chroma_weight_l0[ANV_H264_MAX_REF_FRAMES][2];
   int16_t chroma_offset_l0[ANV_H264_MAX_REF_FRAMES][2];

   int16_t luma_weight_l1[ANV_H264_MAX_REF_FRAMES];
   int16_t luma_offset_l1[ANV_H264_MAX_REF_FRAMES];
   int16_t chroma_weight_l1[ANV_H264_MAX_REF_FRAMES][2];
   int16_t chroma_offset_l1[ANV_H264_MAX_REF_FRAMES][2];

   uint32_t first_mb_byte_offset;
   uint32_t first_mb_bit_offset;

   bool parse_ok;
};

struct anv_h264_rbsp {
   const uint8_t *buf;
   const uint8_t *cur;
   const uint8_t *end;
   int bit;
   int zeros;
   bool error;
   int bits_consumed;
};

bool
anv_h264_parse_slice_header(
   const uint8_t *buf,
   uint32_t slice_nalu_offset,
   uint32_t slice_nalu_size,
   const StdVideoH264SequenceParameterSet *sps,
   const StdVideoH264PictureParameterSet *pps,
   const VkVideoDecodeH264PictureInfoKHR *pic_info,
   uint32_t ref_slot_count,
   const VkVideoReferenceSlotInfoKHR *ref_slots,
   struct anv_h264_slice_params *out);

#endif /* ANV_VIDEO_H264_SLICE_H */
