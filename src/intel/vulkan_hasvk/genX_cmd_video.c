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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "util/vl_zscan_data.h"

#include "genxml/gen_macros.h"
#include "genxml/genX_video_pack.h"

#include "anv_video_h264_slice.h"

static uint32_t
anv_h264_find_slice_nal(const uint8_t *buf, uint32_t off, uint32_t limit)
{
   for (int pass = 0; pass < 2; pass++) {
      if (off + 2 >= limit)
         break;

      uint32_t sc;
      if (buf[off] == 0 && buf[off + 1] == 0 && buf[off + 2] == 1)
         sc = 3;
      else if (off + 3 < limit &&
               buf[off] == 0 && buf[off + 1] == 0 &&
               buf[off + 2] == 0 && buf[off + 3] == 1)
         sc = 4;
      else
         break;

      if (off + sc >= limit)
         break;

      /* Skip AUD and continue to scan. */
      if ((buf[off + sc] & 0x1f) == 9) {
         off += sc + 1;
         while (off + 2 < limit) {
            if (buf[off] == 0 && buf[off + 1] == 0 &&
                (buf[off + 2] == 1 ||
                 (off + 3 < limit &&
                  buf[off + 2] == 0 && buf[off + 3] == 1)))
               break;
            off++;
         }
         continue;
      }

      return off + sc;
   }

   return off + 3;
}

/* With chroma_format_idc == 0 (monochrome) the luma is correct but the chroma
 * is never written so fill it with 0x80 to avoid sampled garbage.
 */
static void
anv_video_init_neutral_chroma(struct anv_device *device,
                              const struct anv_image *img)
{
   struct anv_image *mut_img = (struct anv_image *)img;

   if (img->n_planes < 2 || mut_img->vid_neutral_chroma_init)
      return;

   const struct anv_image_memory_range *range =
      &img->planes[1].primary_surface.memory_range;
   struct anv_address addr = anv_image_address(img, range);
   if (addr.bo == NULL)
      return;

   void *map = addr.bo->map;
   bool mapped_here = false;
   if (!map) {
      if (anv_device_map_bo(device, addr.bo, 0, addr.bo->size,
                            0, &map) != VK_SUCCESS)
         return;
      mapped_here = true;
   }

   memset((uint8_t *)map + addr.offset, 0x80, range->size);

   if (mapped_here)
      anv_device_unmap_bo(device, addr.bo, map, addr.bo->size);

   mut_img->vid_neutral_chroma_init = true;
}

void
genX(CmdBeginVideoCodingKHR) (VkCommandBuffer commandBuffer,
                              const VkVideoBeginCodingInfoKHR * pBeginInfo)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_video_session, vid, pBeginInfo->videoSession);
   ANV_FROM_HANDLE(anv_video_session_params, params,
                   pBeginInfo->videoSessionParameters);

   cmd_buffer->video.vid = vid;
   cmd_buffer->video.params = params;
}

void
genX(CmdControlVideoCodingKHR) (VkCommandBuffer commandBuffer,
                                const VkVideoCodingControlInfoKHR *
                                pCodingControlInfo)
{

}

void genX(CmdEndVideoCodingKHR) (VkCommandBuffer commandBuffer,
                                 const VkVideoEndCodingInfoKHR *
                                 pEndCodingInfo)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

#if GFX_VER == 8
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), flush) {
      flush.PostSyncOperation = 0;
   }
#elif GFX_VER <= 75
   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      pc.CommandStreamerStallEnable = 1;
      pc.StallAtPixelScoreboard = 1;
   }
   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), wait) {
      wait.MFXSyncControlFlag = 1;
   }
#endif

   cmd_buffer->video.vid = NULL;
   cmd_buffer->video.params = NULL;
}

static inline void
anv_h264_sane_field_pocs(const int32_t poc[2], int32_t *top, int32_t *bottom)
{
   int32_t t = poc[0], b = poc[1];
   if (t == INT32_MAX) t = b;
   if (b == INT32_MAX) b = t;
   *top = t; *bottom = b;
}

static void
anv_h264_decode_video(struct anv_cmd_buffer *cmd_buffer,
                      const VkVideoDecodeInfoKHR *frame_info)
{
   /*
    * Build a remapping table from Vulkan DPB slot index to hardware slot index
    * (0-15). Some streams assign DPB slot indices >= 16 but the hardware only
    * supports 16 frame-store entries, so we transparently redirect high slot
    * indices onto unused low ones.
    */
   int8_t slot_to_hw[64];
   memset(slot_to_hw, -1, sizeof(slot_to_hw));
   {
      bool used[16] = {};
      for (uint32_t i = 0; i < frame_info->referenceSlotCount; i++) {
         int s = frame_info->pReferenceSlots[i].slotIndex;
         if (s >= 0 && s < 16) {
            used[s] = true;
            slot_to_hw[s] = (int8_t)s;
         }
      }
      int next_free = 0;
      for (uint32_t i = 0; i < frame_info->referenceSlotCount; i++) {
         int s = frame_info->pReferenceSlots[i].slotIndex;
         if (s >= 16 && s < 64) {
            while (next_free < 16 && used[next_free])
               next_free++;
            if (next_free < 16) {
               slot_to_hw[s] = (int8_t)next_free;
               used[next_free++] = true;
            } else {
               mesa_logw_once("Too many simultaneous H.264 DPB references - dropping.");
            }
         }
      }
   }

   ANV_FROM_HANDLE(anv_buffer, src_buffer, frame_info->srcBuffer);
   struct anv_video_session *vid = cmd_buffer->video.vid;
   struct anv_video_session_params *params = cmd_buffer->video.params;
   const struct VkVideoDecodeH264PictureInfoKHR *h264_pic_info =
      vk_find_struct_const(frame_info->pNext, VIDEO_DECODE_H264_PICTURE_INFO_KHR);
   const StdVideoH264SequenceParameterSet *sps;
   const StdVideoH264PictureParameterSet *pps;
   vk_video_get_h264_parameters(&vid->vk, params ? &params->vk : NULL, frame_info, h264_pic_info, &sps, &pps);

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), flush) {
      flush.VideoPipelineCacheInvalidate = 1;
   };

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_PIPE_MODE_SELECT), sel) {
      sel.StandardSelect = SS_AVC;
      sel.CodecSelect = Decode;
      sel.DecoderShortFormatMode = LongFormatDriverInterface;
      sel.PreDeblockingOutputEnable = false;
      sel.PostDeblockingOutputEnable = true;
      sel.DecoderModeSelect = VLDMode;
   }

   const struct anv_image_view *iv = anv_image_view_from_handle(frame_info->dstPictureResource.imageViewBinding);
   const struct anv_image *img = iv->image;

   if (ANV_DEBUG(VIDEO_DEBUG)) {
      const StdVideoDecodeH264PictureInfo *p = h264_pic_info->pStdPictureInfo;
      mesa_logi("f_num=%d poc=%d/%d field=%u bot=%u idr=%u ref=%u "
                "slices=%u refs=%u dst_layer=%u img=%p layers=%u",
                p->frame_num, p->PicOrderCnt[0], p->PicOrderCnt[1],
                p->flags.field_pic_flag, p->flags.bottom_field_flag,
                p->flags.IdrPicFlag, p->flags.is_reference,
                h264_pic_info->sliceCount, frame_info->referenceSlotCount,
                frame_info->dstPictureResource.baseArrayLayer,
                (void *)img, img->vk.array_layers);
      for (uint32_t i = 0; i < frame_info->referenceSlotCount; i++) {
         const struct anv_image_view *r =
            anv_image_view_from_handle(frame_info->pReferenceSlots[i].pPictureResource->imageViewBinding);
         mesa_logi("ref[%u] slot=%d layer=%u img=%p same_as_dst=%d",
                   i, frame_info->pReferenceSlots[i].slotIndex,
                   frame_info->pReferenceSlots[i].pPictureResource->baseArrayLayer,
                   (void *)r->image, r->image == img);
      }
   }

   /* Conceal as gray in the "non-existing" case. */
   anv_video_init_neutral_chroma(cmd_buffer->device, img);
   for (uint32_t i = 0; i < frame_info->referenceSlotCount; i++) {
      const struct anv_image_view *ref_iv =
         anv_image_view_from_handle(frame_info->pReferenceSlots[i].pPictureResource->imageViewBinding);
      anv_video_init_neutral_chroma(cmd_buffer->device, ref_iv->image);
   }

   uint32_t frame_width_mbs  = sps->pic_width_in_mbs_minus1 + 1;
   uint32_t frame_height_mbs = sps->pic_height_in_map_units_minus1 + 1;
   if (!sps->flags.frame_mbs_only_flag)
      frame_height_mbs *= 2;
   if (frame_width_mbs > 256 || frame_height_mbs > 255)
      mesa_logw_once("Frame %ux%u MBs exceeds hardware limits - clamping.",
                     frame_width_mbs, frame_height_mbs);
   frame_width_mbs  = MIN2(frame_width_mbs,  256u);
   frame_height_mbs = MIN2(frame_height_mbs, 255u);

   uint32_t y_cb_offset =
      (img->planes[1].primary_surface.memory_range.offset -
       img->planes[0].primary_surface.memory_range.offset) /
       img->planes[0].primary_surface.isl.row_pitch_B;

   if (ANV_DEBUG(VIDEO_DEBUG)) {
       mesa_logi("dst extent=%ux%u tiling=%u pitch=%u "
                 "plane0_off=%"PRIu64" plane1_off=%"PRIu64" phys_h=%u y_cb_offset=%u\n",
                 img->vk.extent.width, img->vk.extent.height,
                 img->planes[0].primary_surface.isl.tiling,
                 img->planes[0].primary_surface.isl.row_pitch_B,
                 img->planes[0].primary_surface.memory_range.offset,
                 img->planes[1].primary_surface.memory_range.offset,
                 img->planes[0].primary_surface.isl.phys_level0_sa.h,
                 y_cb_offset);
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_SURFACE_STATE), ss) {
      ss.Width = img->vk.extent.width - 1;
      ss.Height = frame_height_mbs * 16 - 1;
      ss.SurfaceFormat = PLANAR_420_8;
      ss.InterleaveChroma = 1;
      ss.SurfacePitch = img->planes[0].primary_surface.isl.row_pitch_B - 1;
      ss.TiledSurface = img->planes[0].primary_surface.isl.tiling != ISL_TILING_LINEAR;
      ss.TileWalk = TW_YMAJOR;
      ss.YOffsetforUCb = MIN2(y_cb_offset, 32767u);
      ss.YOffsetforVCr = 0;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_PIPE_BUF_ADDR_STATE), buf) {
      memset(buf.ReferencePictureAddress, 0, sizeof(buf.ReferencePictureAddress));
#if GFX_VERx10 == 70
      memset(buf.ReferencePictureArbitrationPriorityControl, 0, sizeof(buf.ReferencePictureArbitrationPriorityControl));
      memset(buf.ReferencePictureGraphicsDataType, 0, sizeof(buf.ReferencePictureGraphicsDataType));
      memset(buf.ReferencePictureCacheabilityControl, 0, sizeof(buf.ReferencePictureCacheabilityControl));
#elif GFX_VERx10 == 75
      memset(buf.ReferencePictureMOCS, 0, sizeof(buf.ReferencePictureMOCS));
#endif
      memset(&buf, 0, sizeof(buf));
      buf.DWordLength = GENX(MFX_PIPE_BUF_ADDR_STATE_length) - 2;
      buf.SubOpcodeB = 2;
      buf.Pipeline = 2;
      buf.CommandType = 3;

      struct anv_address dst_addr =
         anv_image_dpb_address(iv, frame_info->dstPictureResource.baseArrayLayer);
#if GFX_VERx10 == 70
      buf.PostDeblockingDestinationAddress = dst_addr;
      buf.PostDeblockingDestinationCacheabilityControl = 2;
#else
      buf.PostDeblockingDestinationAddress = dst_addr;
      buf.PreDeblockingDestinationMOCS = anv_mocs(cmd_buffer->device, buf.PreDeblockingDestinationAddress.bo, 0);
      buf.PostDeblockingDestinationMOCS = anv_mocs(cmd_buffer->device, buf.PostDeblockingDestinationAddress.bo, 0);
      buf.OriginalUncompressedPictureSourceMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      buf.StreamOutDataDestinationMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
#endif

#if GFX_VER == 8
      buf.IntraRowStoreScratchBufferAddressHigh = (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_INTRA_ROW_STORE].mem->bo,
         vid->vid_mem[ANV_VID_MEM_H264_INTRA_ROW_STORE].offset };
      buf.IntraRowStoreScratchBufferMOCS = anv_mocs(cmd_buffer->device, buf.IntraRowStoreScratchBufferAddressHigh.bo, 0);
      buf.DeblockingFilterRowStoreScratchAddressHigh = (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].mem->bo,
         vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].offset };
      buf.DeblockingFilterRowStoreScratchMOCS = anv_mocs(cmd_buffer->device, buf.DeblockingFilterRowStoreScratchAddressHigh.bo, 0);
      buf.MBStatusBufferMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      buf.MBILDBStreamOutBufferMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
#else
      buf.IntraRowStoreScratchBufferAddress = (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_INTRA_ROW_STORE].mem->bo,
                                                                     vid->vid_mem[ANV_VID_MEM_H264_INTRA_ROW_STORE].offset };
#if GFX_VERx10 == 70
      buf.IntraRowStoreScratchBufferCacheabilityControl = 2;
#elif GFX_VERx10 >= 75
      buf.IntraRowStoreScratchBufferMOCS = anv_mocs(cmd_buffer->device, vid->vid_mem[ANV_VID_MEM_H264_INTRA_ROW_STORE].mem->bo, 0);
#endif
#if GFX_VERx10 == 70
      buf.DeblockingFilterRowStoreScratchBufferAddress =
         (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].mem->bo, vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].offset };
      buf.DeblockingFilterRowStoreScratchBufferCacheabilityControl = 2;
#else
      buf.DeblockingFilterRowStoreScratchAddress =
         (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].mem->bo, vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].offset };
      buf.DeblockingFilterRowStoreScratchMOCS = anv_mocs(cmd_buffer->device, vid->vid_mem[ANV_VID_MEM_H264_DEBLOCK_FILTER_ROW_STORE].mem->bo, 0);
      buf.MBStatusBufferMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      buf.MBILDBStreamOutBufferMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
#endif
#endif

#if GFX_VERx10 == 80
      struct anv_bo *ref_bo = NULL;
#endif
      for (unsigned i = 0; i < frame_info->referenceSlotCount; i++) {
         int s = frame_info->pReferenceSlots[i].slotIndex;
         int hw = (s >= 0 && s < 64) ? (int)slot_to_hw[s] : -1;
         if (hw < 0)
            continue;
         const struct anv_image_view *ref_iv = anv_image_view_from_handle(frame_info->pReferenceSlots[i].pPictureResource->imageViewBinding);
         struct anv_address ref_addr =
            anv_image_dpb_address(ref_iv,
                                  frame_info->pReferenceSlots[i].pPictureResource->baseArrayLayer);
         buf.ReferencePictureAddress[hw] = ref_addr;

#if GFX_VERx10 == 70
         buf.ReferencePictureCacheabilityControl[hw] = 2;
#elif GFX_VERx10 == 75
         buf.ReferencePictureMOCS[hw] = anv_mocs(cmd_buffer->device, ref_addr.bo, 0);
#endif
#if GFX_VERx10 == 80
         if (i == 0)
            ref_bo = ref_iv->image->bindings[0].address.bo;
#endif
      }
#if GFX_VERx10 == 80
      buf.ReferencePictureMOCS = anv_mocs(cmd_buffer->device, ref_bo, 0);
#endif
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_IND_OBJ_BASE_ADDR_STATE), index_obj) {
      index_obj.MFXIndirectBitstreamObjectAddress = anv_address_add(src_buffer->address,
                                                                    frame_info->srcBufferOffset & ~4095);
#if GFX_VERx10 == 70
      index_obj.MFXIndirectBitstreamObjectCacheabilityControl = 2;
#elif GFX_VERx10 == 75
      index_obj.MFXIndirectBitstreamObjectMOCS = anv_mocs(cmd_buffer->device, src_buffer->address.bo,
                                                          0);
      index_obj.MFXIndirectMVObjectMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      index_obj.MFDIndirectITCOEFFObjectMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      index_obj.MFDIndirectITDBLKObjectMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
      index_obj.MFCIndirectPAKBSEObjectMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
#endif
#if GFX_VER == 7
      index_obj.MFXIndirectBitstreamObjectAccessUpperBound = (struct anv_address) { NULL, 0x80000000 };
#endif
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_BSP_BUF_BASE_ADDR_STATE), bsp) {
      bsp.BSDMPCRowStoreScratchBufferAddress = (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_BSD_MPC_ROW_SCRATCH].mem->bo,
         vid->vid_mem[ANV_VID_MEM_H264_BSD_MPC_ROW_SCRATCH].offset };
#if GFX_VERx10 == 70
      bsp.BSDMPCRowStoreScratchBufferCacheabilityControl = 2;
#elif GFX_VERx10 == 75
      bsp.BSDMPCRowStoreScratchBufferMOCS = anv_mocs(cmd_buffer->device, vid->vid_mem[ANV_VID_MEM_H264_BSD_MPC_ROW_SCRATCH].mem->bo, 0);
#endif

      bsp.MPRRowStoreScratchBufferAddress = (struct anv_address) { vid->vid_mem[ANV_VID_MEM_H264_MPR_ROW_SCRATCH].mem->bo,
         vid->vid_mem[ANV_VID_MEM_H264_MPR_ROW_SCRATCH].offset };

#if GFX_VERx10 == 70
      bsp.MPRRowStoreScratchBufferCacheabilityControl = 2;
#elif GFX_VERx10 == 75
      bsp.MPRRowStoreScratchBufferMOCS = anv_mocs(cmd_buffer->device,  vid->vid_mem[ANV_VID_MEM_H264_MPR_ROW_SCRATCH].mem->bo, 0);
      bsp.BitplaneReadBufferMOCS = anv_mocs(cmd_buffer->device, NULL, 0);
#endif
   }

#if GFX_VERx10 >= 75
   anv_batch_emit(&cmd_buffer->batch, GENX(MFD_AVC_PICID_STATE), picid) {
      picid.PictureIDRemappingDisable = true;
   }
#endif

   if (pps->chroma_qp_index_offset < -12 || pps->chroma_qp_index_offset > 12 ||
       pps->second_chroma_qp_index_offset < -12 ||
       pps->second_chroma_qp_index_offset > 12 ||
       pps->pic_init_qp_minus26 < -26 || pps->pic_init_qp_minus26 > 25 ||
       sps->log2_max_frame_num_minus4 > 12 ||
       sps->log2_max_pic_order_cnt_lsb_minus4 > 12) {
      mesa_logw_once("Out of range H.264 SPS/PPS parameter "
                     "(sps_id=%u pps_id=%u chroma_qp=%d/%d init_qp_m26=%d "
                     "log2_frame=%u log2_poc=%u) - clamping.",
                     pps->seq_parameter_set_id, pps->pic_parameter_set_id,
                     pps->chroma_qp_index_offset,
                     pps->second_chroma_qp_index_offset,
                     pps->pic_init_qp_minus26, sps->log2_max_frame_num_minus4,
                     sps->log2_max_pic_order_cnt_lsb_minus4);
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_IMG_STATE), avc_img) {
      avc_img.FrameWidth = MIN2(frame_width_mbs - 1, 255u);
      avc_img.FrameHeight = MIN2(frame_height_mbs - 1, 254u);
      avc_img.FrameSize = frame_width_mbs * frame_height_mbs;

      if (!h264_pic_info->pStdPictureInfo->flags.field_pic_flag)
         avc_img.ImageStructure = FramePicture;
      else if (h264_pic_info->pStdPictureInfo->flags.bottom_field_flag)
         avc_img.ImageStructure = BottomFieldPicture;
      else
         avc_img.ImageStructure = TopFieldPicture;

      avc_img.WeightedBiPredictionIDC = pps->weighted_bipred_idc;
      avc_img.WeightedPredictionEnable = pps->flags.weighted_pred_flag;
      avc_img.FirstChromaQPOffset = CLAMP(pps->chroma_qp_index_offset, -12, 12);
      avc_img.SecondChromaQPOffset = CLAMP(pps->second_chroma_qp_index_offset, -12, 12);
      avc_img.FieldPicture = h264_pic_info->pStdPictureInfo->flags.field_pic_flag;
      avc_img.MBAFFMode = (sps->flags.mb_adaptive_frame_field_flag &&
                           !h264_pic_info->pStdPictureInfo->flags.field_pic_flag);
      avc_img.FrameMBOnly = sps->flags.frame_mbs_only_flag;
      avc_img._8x8IDCTTransformMode = pps->flags.transform_8x8_mode_flag;
      avc_img.Direct8x8Inference = sps->flags.direct_8x8_inference_flag;
      avc_img.ConstrainedIntraPrediction = pps->flags.constrained_intra_pred_flag;
      avc_img.NonReferencePicture = !h264_pic_info->pStdPictureInfo->flags.is_reference;
      avc_img.EntropyCodingSyncEnable = pps->flags.entropy_coding_mode_flag;
      avc_img.ChromaFormatIDC = sps->chroma_format_idc;
      avc_img.TrellisQuantizationChromaDisable = true;
      avc_img.NumberofReferenceFrames = MIN2(frame_info->referenceSlotCount, 16u);
      avc_img.NumberofActiveReferencePicturesfromL0 = pps->num_ref_idx_l0_default_active_minus1 + 1;
      avc_img.NumberofActiveReferencePicturesfromL1 = pps->num_ref_idx_l1_default_active_minus1 + 1;
      avc_img.InitialQPValue = CLAMP(pps->pic_init_qp_minus26, -26, 25);
      avc_img.PicOrderPresent = pps->flags.bottom_field_pic_order_in_frame_present_flag;
      avc_img.DeltaPicOrderAlwaysZero = sps->flags.delta_pic_order_always_zero_flag;
      avc_img.PicOrderCountType = sps->pic_order_cnt_type;
      avc_img.DeblockingFilterControlPresent = pps->flags.deblocking_filter_control_present_flag;
      avc_img.RedundantPicCountPresent = pps->flags.redundant_pic_cnt_present_flag;
      avc_img.Log2MaxFrameNumber = MIN2((uint32_t)sps->log2_max_frame_num_minus4, 12u);
      avc_img.Log2MaxPicOrderCountLSB = MIN2((uint32_t)sps->log2_max_pic_order_cnt_lsb_minus4, 12u);
      avc_img.CurrentPictureFrameNumber = h264_pic_info->pStdPictureInfo->frame_num;
   }

   if (ANV_DEBUG(VIDEO_DEBUG) &&
       h264_pic_info->pStdPictureInfo->flags.field_pic_flag) {
      const StdVideoDecodeH264PictureInfo *p = h264_pic_info->pStdPictureInfo;
      mesa_logi("field_pic=%u bottom=%u mb_adaptive=%u frame_mbs_only=%u "
                "mbaff_frame_flag=%u width_mbs=%u height_mbs=%u chroma_idc=%u entropy=%u "
                "8x8=%u chroma_qp=%d/%d init_qp_m26=%d wp=%u wbi=%u frame_num=%u poc=%d/%d "
                "constr_intra=%u num_ref=%u nact_l0=%u nact_l1=%u",
                p->flags.field_pic_flag, p->flags.bottom_field_flag,
                sps->flags.mb_adaptive_frame_field_flag, sps->flags.frame_mbs_only_flag,
                (unsigned)(sps->flags.mb_adaptive_frame_field_flag && !p->flags.field_pic_flag),
                frame_width_mbs, frame_height_mbs, sps->chroma_format_idc,
                pps->flags.entropy_coding_mode_flag, pps->flags.transform_8x8_mode_flag,
                pps->chroma_qp_index_offset, pps->second_chroma_qp_index_offset,
                pps->pic_init_qp_minus26, pps->flags.weighted_pred_flag,
                pps->weighted_bipred_idc, p->frame_num,
                p->PicOrderCnt[0], p->PicOrderCnt[1],
                pps->flags.constrained_intra_pred_flag,
                frame_info->referenceSlotCount,
                pps->num_ref_idx_l0_default_active_minus1 + 1,
                pps->num_ref_idx_l1_default_active_minus1 + 1);
   }

   StdVideoH264ScalingLists scaling_lists;
   vk_video_derive_h264_scaling_list(sps, pps, &scaling_lists);
   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_QM_STATE), qm) {
      qm.DWordLength = 16;
      qm.AVC = AVC_4x4_Intra_MATRIX;
      for (unsigned m = 0; m < 3; m++)
         for (unsigned q = 0; q < 16; q++)
            qm.ForwardQuantizerMatrix[m * 16 + vl_zscan_normal_16[q]] =
               scaling_lists.ScalingList4x4[m][q];
   }
   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_QM_STATE), qm) {
      qm.DWordLength = 16;
      qm.AVC = AVC_4x4_Inter_MATRIX;
      for (unsigned m = 0; m < 3; m++)
         for (unsigned q = 0; q < 16; q++)
            qm.ForwardQuantizerMatrix[m * 16 + vl_zscan_normal_16[q]] =
               scaling_lists.ScalingList4x4[m + 3][q];
   }
   if (pps->flags.transform_8x8_mode_flag) {
      anv_batch_emit(&cmd_buffer->batch, GENX(MFX_QM_STATE), qm) {
         qm.DWordLength = 16;
         qm.AVC = AVC_8x8_Intra_MATRIX;
         for (unsigned q = 0; q < 64; q++)
            qm.ForwardQuantizerMatrix[vl_zscan_normal[q]] =
               scaling_lists.ScalingList8x8[0][q];
      }
      anv_batch_emit(&cmd_buffer->batch, GENX(MFX_QM_STATE), qm) {
         qm.DWordLength = 16;
         qm.AVC = AVC_8x8_Inter_MATRIX;
         for (unsigned q = 0; q < 64; q++)
            qm.ForwardQuantizerMatrix[vl_zscan_normal[q]] =
               scaling_lists.ScalingList8x8[1][q];
      }
   }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
   struct anv_bo *dmv_bo = NULL;
#pragma GCC diagnostic pop

   const uint8_t *bs_map = src_buffer->address.bo->map;
   bool bs_mapped_here = false;
   if (!bs_map) {
      void *tmp = NULL;
      if (anv_device_map_bo(cmd_buffer->device,
                            src_buffer->address.bo,
                            0, src_buffer->address.bo->size,
                            0, &tmp) == VK_SUCCESS) {
         bs_map = (const uint8_t *)tmp;
         bs_mapped_here = true;
      }
   }

   uint32_t mb_width = sps->pic_width_in_mbs_minus1 + 1;
   uint32_t pic_height_in_map_units = sps->pic_height_in_map_units_minus1 + 1;
   uint32_t mb_height = sps->flags.frame_mbs_only_flag ?
                        pic_height_in_map_units : pic_height_in_map_units * 2;

   /* Slice vertical positions are expressed in macroblock rows.
    * Under MBAFF first_mb_in_slice addresses a macroblock pair, so the true MB
    * row is (first_mb_in_slice / mb_width) * 2 while the column is unchanged.
    */
   bool field_pic = h264_pic_info->pStdPictureInfo->flags.field_pic_flag;
   bool mbaff = sps->flags.mb_adaptive_frame_field_flag && !field_pic;
   uint32_t pic_height_mbs = field_pic ? (mb_height / 2) : mb_height;

   struct anv_h264_slice_params *sp =
      calloc(h264_pic_info->sliceCount, sizeof(*sp));
   if (sp) {
      for (unsigned s = 0; s < h264_pic_info->sliceCount; s++) {
         uint32_t slice_off = h264_pic_info->pSliceOffsets[s];
         uint32_t buf_end = (s + 1 < h264_pic_info->sliceCount) ?
                             h264_pic_info->pSliceOffsets[s + 1] :
                             frame_info->srcBufferRange;

         uint32_t bo_base = frame_info->srcBufferOffset & ~4095u;
         uint32_t page_off = frame_info->srcBufferOffset & 4095u;

         uint32_t bo_nalu = bs_map ?
            anv_h264_find_slice_nal(bs_map + bo_base,
                                    page_off + slice_off,
                                    page_off + buf_end) :
            page_off + slice_off + 3;
         uint32_t nalu_size = (page_off + buf_end > bo_nalu) ?
                              (page_off + buf_end - bo_nalu) : 0;

         bool ok = false;
         if (bs_map && nalu_size > 0 &&
             bo_base + bo_nalu + nalu_size <= src_buffer->address.bo->size) {
            ok = anv_h264_parse_slice_header(
                    bs_map + bo_base, bo_nalu, nalu_size,
                    sps, pps, h264_pic_info,
                    frame_info->referenceSlotCount,
                    frame_info->pReferenceSlots,
                    &sp[s]);
         }
         if (!ok) {
            memset(&sp[s], 0, sizeof(sp[s]));
            sp[s].slice_type = ANV_H264_SLICE_I;
            sp[s].first_mb_in_slice = 0;
            sp[s].num_ref_idx_l0_active_minus1 =
               pps->num_ref_idx_l0_default_active_minus1;
            sp[s].num_ref_idx_l1_active_minus1 =
               pps->num_ref_idx_l1_default_active_minus1;
            memset(sp[s].ref_list0, ANV_H264_INVALID_SLOT,
                   sizeof(sp[s].ref_list0));
            memset(sp[s].ref_list1, ANV_H264_INVALID_SLOT,
                   sizeof(sp[s].ref_list1));
         }
         sp[s].parse_ok = ok;
      }
   }

   uint32_t buffer_offset = frame_info->srcBufferOffset & 4095;
#define HEADER_OFFSET 3
   for (unsigned s = 0; s < h264_pic_info->sliceCount; s++) {
      bool last_slice = s == (h264_pic_info->sliceCount - 1);
      uint32_t current_offset = h264_pic_info->pSliceOffsets[s];
      uint32_t this_end;
      if (!last_slice) {
         this_end = h264_pic_info->pSliceOffsets[s + 1];
      } else {
         this_end = frame_info->srcBufferRange;
      }

      anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_DIRECTMODE_STATE), avc_directmode) {
         for (unsigned i = 0; i < frame_info->referenceSlotCount; i++) {
            int s = frame_info->pReferenceSlots[i].slotIndex;
            int hw = (s >= 0 && s < 64) ? (int)slot_to_hw[s] : -1;
            if (hw < 0)
               continue;
            const struct VkVideoDecodeH264DpbSlotInfoKHR *dpb_slot =
               vk_find_struct_const(frame_info->pReferenceSlots[i].pNext, VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR);
            const struct anv_image_view *ref_iv = anv_image_view_from_handle(frame_info->pReferenceSlots[i].pPictureResource->imageViewBinding);
            const StdVideoDecodeH264ReferenceInfo *ref_info = dpb_slot->pStdReferenceInfo;
            uint32_t ref_layer =
               frame_info->pReferenceSlots[i].pPictureResource->baseArrayLayer;
            if (i == 0)
               dmv_bo = anv_image_dmv_top_address(ref_iv, ref_layer).bo;
#if GFX_VERx10 == 70
            {
               bool ref_is_field = ref_info &&
                  (ref_info->flags.top_field_flag ^ ref_info->flags.bottom_field_flag);
               bool ref_dmv_bottom = ref_is_field &&
                  !sps->flags.direct_8x8_inference_flag;
               avc_directmode.DirectMVBufferAddress[hw * 2] =
                  anv_image_dmv_top_address(ref_iv, ref_layer);
               avc_directmode.DirectMVBufferAddress[hw * 2 + 1] = ref_dmv_bottom ?
                  anv_image_dmv_bottom_address(ref_iv, ref_layer) :
                  anv_image_dmv_top_address(ref_iv, ref_layer);
            }
            avc_directmode.DirectMVBufferCacheabilityControl[hw * 2] = 2;
            avc_directmode.DirectMVBufferCacheabilityControl[hw * 2 + 1] = 2;
#elif GFX_VERx10 == 75
            {
               struct anv_address dmv_addr = anv_image_dmv_top_address(ref_iv, ref_layer);
               if (hw == 0) {
                  avc_directmode.DirectMVBuffer0Address = dmv_addr;
                  avc_directmode.DirectMVBufferMOCS = anv_mocs(cmd_buffer->device, dmv_addr.bo, 0);
               } else {
                  avc_directmode.DirectMVBufferAddress1[hw - 1] = dmv_addr;
                  avc_directmode.DirectMVBufferMOCS1[hw - 1] = anv_mocs(cmd_buffer->device, dmv_addr.bo, 0);
               }
            }
#elif GFX_VERx10 == 80
            avc_directmode.DirectMVBufferAddress[hw] = anv_image_dmv_top_address(ref_iv, ref_layer);
#endif
            {
               int32_t ref_top, ref_bot;
               anv_h264_sane_field_pocs(ref_info->PicOrderCnt, &ref_top, &ref_bot);
               avc_directmode.POCList[2 * hw] = ref_top;
               avc_directmode.POCList[2 * hw + 1] = ref_bot;
            }
         }
         uint32_t dst_layer = frame_info->dstPictureResource.baseArrayLayer;
#if GFX_VERx10 == 70
         {
            bool cur_dmv_bottom = field_pic && !sps->flags.direct_8x8_inference_flag;
            avc_directmode.DirectMVBufferWriteAddress[0] =
               anv_image_dmv_top_address(iv, dst_layer);
            avc_directmode.DirectMVBufferWriteAddress[1] = cur_dmv_bottom ?
               anv_image_dmv_bottom_address(iv, dst_layer) :
               anv_image_dmv_top_address(iv, dst_layer);
         }
         avc_directmode.DirectMVBufferWriteCacheabilityControl[0] = 2;
         avc_directmode.DirectMVBufferWriteCacheabilityControl[1] = 2;
#elif GFX_VERx10 == 75
         avc_directmode.DirectMVBufferWriteAddress = anv_image_dmv_top_address(iv, dst_layer);
         avc_directmode.DirectMVBufferWriteMOCS = anv_mocs(cmd_buffer->device, avc_directmode.DirectMVBufferWriteAddress.bo, 0);
         if (!avc_directmode.DirectMVBufferMOCS)
            avc_directmode.DirectMVBufferMOCS = anv_mocs(cmd_buffer->device, avc_directmode.DirectMVBufferWriteAddress.bo, 0);
#elif GFX_VERx10 == 80
         avc_directmode.DirectMVBufferAttributes.TargetCache = 2; /* LLC/eLLC */
         avc_directmode.DirectMVBufferWriteAddress = anv_image_dmv_top_address(iv, dst_layer);
         avc_directmode.DirectMVBufferWriteAttributes.TargetCache = 2;
#endif
         {
            int32_t cur_top, cur_bot;
            anv_h264_sane_field_pocs(h264_pic_info->pStdPictureInfo->PicOrderCnt,
                                     &cur_top, &cur_bot);
            avc_directmode.POCList[32] = cur_top;
            avc_directmode.POCList[33] = cur_bot;
         }
      }

      const struct anv_h264_slice_params *cur = sp ? &sp[s] : NULL;

      for (unsigned list = 0; list <= 1; list++) {
         anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_REF_IDX_STATE), ref_idx) {
            ref_idx.ReferencePictureListSelect = list;
            const uint8_t *rl = (list == 0) ?
               (cur ? cur->ref_list0 : NULL) :
               (cur ? cur->ref_list1 : NULL);
            const uint8_t *rlb = (list == 0) ?
               (cur ? cur->ref_list0_bottom : NULL) :
               (cur ? cur->ref_list1_bottom : NULL);
            for (unsigned e = 0; e < 32; e++) {
               uint8_t slot = (rl && e < ANV_H264_MAX_REF_FRAMES) ?
                              rl[e] : ANV_H264_INVALID_SLOT;
               if (slot == ANV_H264_INVALID_SLOT) {
                  ref_idx.ReferenceListEntry[e] = 0xFF;
               } else {
                  int8_t hw = (slot < 64) ? slot_to_hw[slot] : -1;
                  if (hw < 0) {
                     ref_idx.ReferenceListEntry[e] = 0xFF;
                  } else {
                     bool is_long = false, ref_top = false, ref_bottom_f = false;
                     for (uint32_t i = 0; i < frame_info->referenceSlotCount; i++) {
                        if (frame_info->pReferenceSlots[i].slotIndex == slot) {
                           const struct VkVideoDecodeH264DpbSlotInfoKHR *d =
                              vk_find_struct_const(frame_info->pReferenceSlots[i].pNext,
                                                   VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR);
                           if (d && d->pStdReferenceInfo) {
                              is_long = d->pStdReferenceInfo->flags.used_for_long_term_reference ||
                                        d->pStdReferenceInfo->flags.is_non_existing;
                              ref_top = d->pStdReferenceInfo->flags.top_field_flag;
                              ref_bottom_f = d->pStdReferenceInfo->flags.bottom_field_flag;
                           }
                           break;
                        }
                     }
                     if (field_pic && rlb && e < ANV_H264_MAX_REF_FRAMES) {
                        ref_bottom_f = rlb[e];
                        ref_top = !rlb[e];
                     }
                     bool ref_is_field = ref_top ^ ref_bottom_f;
                     bool ref_bottom = (!ref_top) & ref_bottom_f;
                     ref_idx.ReferenceListEntry[e] =
                        (is_long ? 0x40u : 0x00u) |
                        (ref_is_field ? 0x00u : 0x20u) |
                        ((uint8_t)hw << 1) |
                        (ref_bottom ? 0x01u : 0x00u);
                  }
               }
            }
         }
      }

      /* Emit MFX_AVC_WEIGHTOFFSET_STATE for both list 0 and list 1. */
      bool is_b = cur && cur->slice_type == ANV_H264_SLICE_B;
      bool need_w_l0 = cur &&
         ((pps->flags.weighted_pred_flag &&
           (cur->slice_type == ANV_H264_SLICE_P ||
            cur->slice_type == ANV_H264_SLICE_SP)) ||
          (pps->weighted_bipred_idc == 1 && is_b));
      bool need_w_l1 = pps->weighted_bipred_idc == 1 && is_b;
      if (need_w_l0 || need_w_l1) {
         for (unsigned list = 0; list <= 1; list++) {
            if (list == 0 && !need_w_l0) continue;
            if (list == 1 && !need_w_l1) continue;
            anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_WEIGHTOFFSET_STATE), wo) {
               wo.WeightandOffsetSelect = list;
               int active = (list == 0) ?
                  cur->num_ref_idx_l0_active_minus1 + 1 :
                  cur->num_ref_idx_l1_active_minus1 + 1;
               if (active > ANV_H264_MAX_REF_FRAMES)
                  active = ANV_H264_MAX_REF_FRAMES;
               for (int i = 0; i < 32; i++) {
                  int16_t lw = (i < active) ? ((list == 0) ? cur->luma_weight_l0[i] : cur->luma_weight_l1[i]) : 0;
                  int16_t lo = (i < active) ? ((list == 0) ? cur->luma_offset_l0[i] : cur->luma_offset_l1[i]) : 0;
                  int16_t cw0 = (i < active) ? ((list == 0) ? cur->chroma_weight_l0[i][0] : cur->chroma_weight_l1[i][0]) : 0;
                  int16_t co0 = (i < active) ? ((list == 0) ? cur->chroma_offset_l0[i][0] : cur->chroma_offset_l1[i][0]) : 0;
                  int16_t cw1 = (i < active) ? ((list == 0) ? cur->chroma_weight_l0[i][1] : cur->chroma_weight_l1[i][1]) : 0;
                  int16_t co1 = (i < active) ? ((list == 0) ? cur->chroma_offset_l0[i][1] : cur->chroma_offset_l1[i][1]) : 0;
                  wo.WeightOffset[i * 3 + 0] = ((uint32_t)(uint16_t)lo  << 16) | (uint16_t)lw;
                  wo.WeightOffset[i * 3 + 1] = ((uint32_t)(uint16_t)co0 << 16) | (uint16_t)cw0;
                  wo.WeightOffset[i * 3 + 2] = ((uint32_t)(uint16_t)co1 << 16) | (uint16_t)cw1;
               }
            }
         }
      }

      /* The "phantom slice" for error concealment. */
      if (s == 0 && cur && cur->first_mb_in_slice != 0) {
         uint32_t first_hor = cur->first_mb_in_slice % mb_width;
         uint32_t first_ver = cur->first_mb_in_slice / mb_width;
         if (mbaff)
            first_ver <<= 1;
         anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_SLICE_STATE), phantom) {
            phantom.SliceType = ISlice;
            phantom.SliceStartMBNumber = 0;
            phantom.SliceHorizontalPosition = 0;
            phantom.SliceVerticalPosition = 0;
            phantom.NextSliceHorizontalPosition = first_hor;
            phantom.NextSliceVerticalPosition = first_ver;
            phantom.SliceQuantizationParameter = (uint32_t)(pps->pic_init_qp_minus26 + 26);
            phantom.LastSliceGroup = false;
         }
         anv_batch_emit(&cmd_buffer->batch, GENX(MFD_AVC_BSD_OBJECT), phantom_bsd) {
            phantom_bsd.IndirectBSDDataLength = 0;
            phantom_bsd.IndirectBSDDataStartAddress = 0;
            phantom_bsd.InlineData.LastSlice = false;
            phantom_bsd.InlineData.FixPrevMBSkipped = 1;
#if GFX_VERx10 >= 75
            phantom_bsd.InlineData.IntraPredictionErrorControl = 1;
            phantom_bsd.InlineData.Intra8x84x4PredictionErrorConcealmentControl = 1;
            phantom_bsd.InlineData.ISliceConcealmentMode = 1;
#endif
         }
      }

      {
         int st = cur ? cur->slice_type : ANV_H264_SLICE_I;
         uint32_t cur_fmb = cur ? (uint32_t)cur->first_mb_in_slice : 0;
         uint32_t cur_hor = cur_fmb % mb_width;
         uint32_t cur_ver = cur_fmb / mb_width;
         if (mbaff)
            cur_ver <<= 1;

         uint32_t next_hor, next_ver;
         if (last_slice || !sp) {
            /* This value is already in MB rows, so it can't be doubled for MBAFF. */
            next_hor = 0;
            next_ver = pic_height_mbs;
         } else {
            uint32_t next_fmb = (uint32_t)sp[s + 1].first_mb_in_slice;
            next_hor = next_fmb % mb_width;
            next_ver = next_fmb / mb_width;
            if (mbaff)
               next_ver <<= 1;
            if (next_ver >= pic_height_mbs) { next_hor = 0; next_ver = pic_height_mbs; }
         }

         anv_batch_emit(&cmd_buffer->batch, GENX(MFX_AVC_SLICE_STATE), ss) {
            ss.SliceType =
               (st == ANV_H264_SLICE_P || st == ANV_H264_SLICE_SP) ? PSlice :
               (st == ANV_H264_SLICE_B)                            ? BSlice :
                                                                   ISlice;

            bool implicit_bipred = (st == ANV_H264_SLICE_B &&
                                    pps->weighted_bipred_idc == 2);
            ss.Log2WeightDenominatorLuma = implicit_bipred ? 5 :
                                           (cur ? CLAMP(cur->luma_log2_weight_denom, 0, 7) : 0);
            ss.Log2WeightDenominatorChroma = implicit_bipred ? 5 :
                                             (cur ? CLAMP(cur->chroma_log2_weight_denom, 0, 7) : 0);
            ss.NumberofReferencePicturesinInterpredictionList0 =
               (cur && st != ANV_H264_SLICE_I && st != ANV_H264_SLICE_SI) ?
               MIN2((uint32_t)(cur->num_ref_idx_l0_active_minus1 + 1),
                    (uint32_t)ANV_H264_MAX_REF_FRAMES) : 0;
            ss.NumberofReferencePicturesinInterpredictionList1 =
               (cur && st == ANV_H264_SLICE_B) ?
               MIN2((uint32_t)(cur->num_ref_idx_l1_active_minus1 + 1),
                    (uint32_t)ANV_H264_MAX_REF_FRAMES) : 0;
            ss.SliceAlphaC0OffsetDiv2 = CLAMP(cur ? cur->slice_alpha_c0_offset_div2 : 0, -6, 6);
            ss.SliceBetaOffsetDiv2 = CLAMP(cur ? cur->slice_beta_offset_div2 : 0, -6, 6);
            ss.SliceQuantizationParameter =
               (uint32_t)CLAMP((int)(pps->pic_init_qp_minus26 + 26 +
               (cur ? cur->slice_qp_delta : 0)), 0, 51);
            ss.CABACInitIDC = cur ? MIN2((uint32_t)cur->cabac_init_idc, 2u) : 0;
            ss.DisableDeblockingFilterIndicator =
               cur ? MIN2((uint32_t)cur->disable_deblocking_filter_idc, 2u) : 0;
            ss.DirectPredictionType =
               (cur && cur->direct_spatial_mv_pred_flag) ? Spatial : Temporal;
            ss.WeightedPredictionIndicator =
               (st == ANV_H264_SLICE_B) ? (uint32_t)pps->weighted_bipred_idc :
               need_w_l0 ? 1u : 0u;
            ss.SliceStartMBNumber = MIN2(cur_fmb, 32767u);
            ss.SliceHorizontalPosition = MIN2(cur_hor, 255u);
            ss.SliceVerticalPosition = MIN2(cur_ver, 255u);
            ss.NextSliceHorizontalPosition = MIN2(next_hor, 255u);
            ss.NextSliceVerticalPosition = MIN2(next_ver, 255u);
            ss.LastSliceGroup = last_slice;
            ss.SliceID = s & (uint32_t)((1u << 4) - 1);
         }

         if (ANV_DEBUG(VIDEO_DEBUG) && field_pic && cur) {
            char l0[128] = {0}, l1[128] = {0};
            for (int e = 0; e < 8; e++) {
               char t[16];
               snprintf(t, sizeof(t), "%d ", (int8_t)cur->ref_list0[e]);
               strncat(l0, t, sizeof(l0) - strlen(l0) - 1);
               snprintf(t, sizeof(t), "%d ", (int8_t)cur->ref_list1[e]);
               strncat(l1, t, sizeof(l1) - strlen(l1) - 1);
            }
            mesa_logi("slice[%u]: type=%d first_mb=%u qp=%d cabac_init=%d "
                      "disable_dblk=%d spatial_direct=%d nref_l0=%d nref_l1=%d "
                      "hor=%u ver=%u next_hor=%u next_ver=%u last=%d | L0: %s| L1: %s",
                      s, st, cur_fmb,
                      (int)(pps->pic_init_qp_minus26 + 26 + cur->slice_qp_delta),
                      cur->cabac_init_idc, cur->disable_deblocking_filter_idc,
                      cur->direct_spatial_mv_pred_flag,
                      cur->num_ref_idx_l0_active_minus1 + 1,
                      cur->num_ref_idx_l1_active_minus1 + 1,
                      cur_hor, cur_ver, next_hor, next_ver, last_slice, l0, l1);
         }
      }

      uint32_t bsd_nal = bs_map ?
         anv_h264_find_slice_nal(
            bs_map + (frame_info->srcBufferOffset & ~4095u),
            buffer_offset + current_offset,
            buffer_offset + this_end) :
         buffer_offset + current_offset + HEADER_OFFSET;
      anv_batch_emit(&cmd_buffer->batch, GENX(MFD_AVC_BSD_OBJECT), avc_bsd) {
         /* Bound the bitstream object to this slice's data. */
         avc_bsd.IndirectBSDDataLength = buffer_offset + this_end - bsd_nal;
         avc_bsd.IndirectBSDDataStartAddress = bsd_nal;
         avc_bsd.InlineData.LastSlice = last_slice;
         avc_bsd.InlineData.FixPrevMBSkipped = 1;
         avc_bsd.InlineData.MBHeaderErrorHandling = 1;
         avc_bsd.InlineData.EntropyErrorHandling = 1;
         avc_bsd.InlineData.MPRErrorHandling = 1;
         avc_bsd.InlineData.BSDPrematureCompleteErrorHandling = 1;
         avc_bsd.InlineData.IntraPredMode4x48x8LumaErrorControl = 1;
         /* Initially I thought Ivy Bridge was incapable of doing short mode
          * and loved to draw rainbows all day, but the more likely scenario
          * was a fundamental breakdown here.
          */
         if (cur) {
            avc_bsd.InlineData.FirstMBByteOffsetofSliceDataorSliceHeader =
               cur->first_mb_byte_offset;
            avc_bsd.InlineData.FirstMBBitOffset = cur->first_mb_bit_offset;
         }
#if GFX_VERx10 >= 75
         avc_bsd.InlineData.IntraPredictionErrorControl = 1;
         avc_bsd.InlineData.Intra8x84x4PredictionErrorConcealmentControl = 1;
         avc_bsd.InlineData.ISliceConcealmentMode = 1;
#endif
      }
   }

   free(sp);
   if (bs_mapped_here)
      anv_device_unmap_bo(cmd_buffer->device,
                          src_buffer->address.bo,
                          (void *)bs_map,
                          src_buffer->address.bo->size);
}

void
genX(CmdDecodeVideoKHR) (VkCommandBuffer commandBuffer,
                         const VkVideoDecodeInfoKHR * frame_info)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   if (!cmd_buffer->video.vid)
      abort();

   switch (cmd_buffer->video.vid->vk.op) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
      anv_h264_decode_video(cmd_buffer, frame_info);
      break;
   default:
      UNREACHABLE("Unsupported video codec operation!");
   }
}
