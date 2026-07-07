/*
 * Copyright © 2026 dancingmirrors@icloud.com
 * Copyright © 2026 irql-notlessorequal
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
#include <stdbool.h>
#include <string.h>

#include "util/macros.h"
#include "anv_video_h264_slice.h"

#define VALID_RBSP_OR_BAIL(st) \
   if (unlikely(st.error)) \
      return false;

static inline void
anv_h264_rbsp_init(struct anv_h264_rbsp *st,
                   const uint8_t *buf, size_t size)
{
   st->buf = buf;
   st->cur = buf;
   st->end = buf + size;
   st->bit = 7;
   st->zeros = 0;
   st->error = false;
   st->bits_consumed = 0;
}

static inline int
anv_h264_rbsp_next_byte(struct anv_h264_rbsp *st)
{
   if (st->cur >= st->end) { st->error = true; return -1; }
   uint8_t c = *st->cur++;
   if (c == 0) {
      st->zeros++;
   } else {
      st->zeros = 0;
   }
   if (st->zeros >= 2 && st->cur < st->end && *st->cur == 0x03) {
      st->cur++;
      st->zeros = 0;
   }
   return c;
}

static inline int
anv_h264_rbsp_consume_bit(struct anv_h264_rbsp *st)
{
   if (st->cur >= st->end) { st->error = true; return 0; }
   int v = !!(*st->cur & (1u << st->bit));
   st->bits_consumed++;
   if (st->bit > 0) {
      st->bit--;
   } else {
      if (anv_h264_rbsp_next_byte(st) < 0) return 0;
      st->bit = 7;
   }
   return v;
}

static inline unsigned
anv_h264_rbsp_u(struct anv_h264_rbsp *st, int n)
{
   unsigned v = 0;
   for (int i = 0; i < n && !st->error; i++)
      v = (v << 1) | anv_h264_rbsp_consume_bit(st);
   return v;
}

static inline unsigned
anv_h264_rbsp_ue(struct anv_h264_rbsp *st)
{
   int leading = 0;
   while (!st->error && !anv_h264_rbsp_consume_bit(st))
      leading++;
   if (leading == 0) return 0;
   return (1u << leading) - 1 + anv_h264_rbsp_u(st, leading);
}

static inline int
anv_h264_rbsp_se(struct anv_h264_rbsp *st)
{
   unsigned v = anv_h264_rbsp_ue(st);
   return (v & 1) ? (int)((v + 1) >> 1) : -(int)(v >> 1);
}

struct anv_h264_fref {
   uint8_t  slot;
   uint8_t  parities;
   int32_t  frame_poc;
   int32_t  fnwrap;
   uint32_t lt_idx;
   bool     is_long;
};

#define ANV_H264_PAR_TOP    1
#define ANV_H264_PAR_BOTTOM 2

static void
anv_h264_gather_frefs(
   uint32_t ref_slot_count,
   const VkVideoReferenceSlotInfoKHR *ref_slots,
   uint32_t cur_frame_num,
   uint32_t MaxFrameNum,
   struct anv_h264_fref *st, int *nst_out,
   struct anv_h264_fref *lt, int *nlt_out)
{
   int nst = 0, nlt = 0;
   for (uint32_t i = 0; i < ref_slot_count; i++) {
      const VkVideoDecodeH264DpbSlotInfoKHR *dpb =
         vk_find_struct_const(ref_slots[i].pNext,
                              VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR);
      if (!dpb || !dpb->pStdReferenceInfo) continue;
      const StdVideoDecodeH264ReferenceInfo *ri = dpb->pStdReferenceInfo;

      uint8_t parities = (ri->flags.top_field_flag ? ANV_H264_PAR_TOP : 0) |
                         (ri->flags.bottom_field_flag ? ANV_H264_PAR_BOTTOM : 0);
      if (parities == 0)
         parities = ANV_H264_PAR_TOP | ANV_H264_PAR_BOTTOM;

      struct anv_h264_fref f;
      f.slot = (uint8_t)ref_slots[i].slotIndex;
      f.parities = parities;
      f.is_long = ri->flags.used_for_long_term_reference;

      int32_t poc_top = ri->PicOrderCnt[0];
      int32_t poc_bot = ri->PicOrderCnt[1];
      if (parities == ANV_H264_PAR_TOP)
         f.frame_poc = poc_top;
      else if (parities == ANV_H264_PAR_BOTTOM)
         f.frame_poc = poc_bot;
      else
         f.frame_poc = MIN2(poc_top, poc_bot);

      f.lt_idx = ri->FrameNum;
      f.fnwrap = (ri->FrameNum > cur_frame_num) ?
                 (int32_t)ri->FrameNum - (int32_t)MaxFrameNum :
                 (int32_t)ri->FrameNum;

      if (f.is_long) {
         if (nlt < ANV_H264_MAX_REF_FRAMES) lt[nlt++] = f;
      } else {
         if (nst < ANV_H264_MAX_REF_FRAMES) st[nst++] = f;
      }
   }
   *nst_out = nst;
   *nlt_out = nlt;
}

/* H.264 8.2.4.2.5 */
static void
anv_h264_emit_field_list(uint8_t *slots, uint8_t *bottoms, int cap, int *idx,
                         const struct anv_h264_fref *in, int len, int sel)
{
   int i0 = 0, i1 = 0;
   int opp = sel ^ 3;
   while ((i0 < len || i1 < len) && *idx < cap) {
      while (i0 < len && !(in[i0].parities & sel)) i0++;
      while (i1 < len && !(in[i1].parities & opp)) i1++;
      if (i0 < len && *idx < cap) {
         slots[*idx] = in[i0].slot;
         bottoms[*idx] = (sel == ANV_H264_PAR_BOTTOM);
         (*idx)++; i0++;
      }
      if (i1 < len && *idx < cap) {
         slots[*idx] = in[i1].slot;
         bottoms[*idx] = (opp == ANV_H264_PAR_BOTTOM);
         (*idx)++; i1++;
      }
   }
}

static void
anv_h264_sort_fnwrap_desc(struct anv_h264_fref *a, int n)
{
   for (int i = 1; i < n; i++) {
      struct anv_h264_fref v = a[i];
      int j = i - 1;
      while (j >= 0 && a[j].fnwrap < v.fnwrap) { a[j+1] = a[j]; j--; }
      a[j+1] = v;
   }
}

static void
anv_h264_sort_ltidx_asc(struct anv_h264_fref *a, int n)
{
   for (int i = 1; i < n; i++) {
      struct anv_h264_fref v = a[i];
      int j = i - 1;
      while (j >= 0 && a[j].lt_idx > v.lt_idx) { a[j+1] = a[j]; j--; }
      a[j+1] = v;
   }
}

static void
anv_h264_build_field_ref_lists(
   int slice_type, int sel, int32_t cur_field_poc,
   uint32_t cur_frame_num, uint32_t MaxFrameNum,
   uint32_t ref_slot_count, const VkVideoReferenceSlotInfoKHR *ref_slots,
   uint8_t l0[], uint8_t l0b[], uint8_t l1[], uint8_t l1b[])
{
   memset(l0, ANV_H264_INVALID_SLOT, ANV_H264_MAX_REF_FRAMES);
   memset(l1, ANV_H264_INVALID_SLOT, ANV_H264_MAX_REF_FRAMES);
   memset(l0b, 0, ANV_H264_MAX_REF_FRAMES);
   memset(l1b, 0, ANV_H264_MAX_REF_FRAMES);

   if (slice_type == ANV_H264_SLICE_I || slice_type == ANV_H264_SLICE_SI)
      return;

   struct anv_h264_fref st[ANV_H264_MAX_REF_FRAMES];
   struct anv_h264_fref lt[ANV_H264_MAX_REF_FRAMES];
   int nst = 0, nlt = 0;
   anv_h264_gather_frefs(ref_slot_count, ref_slots, cur_frame_num, MaxFrameNum,
                         st, &nst, lt, &nlt);

   anv_h264_sort_ltidx_asc(lt, nlt);

   if (slice_type == ANV_H264_SLICE_P || slice_type == ANV_H264_SLICE_SP) {
      anv_h264_sort_fnwrap_desc(st, nst);
      int idx = 0;
      anv_h264_emit_field_list(l0, l0b, ANV_H264_MAX_REF_FRAMES, &idx,
                               st, nst, sel);
      anv_h264_emit_field_list(l0, l0b, ANV_H264_MAX_REF_FRAMES, &idx,
                               lt, nlt, sel);
      return;
   }

   struct anv_h264_fref o0[ANV_H264_MAX_REF_FRAMES];
   struct anv_h264_fref o1[ANV_H264_MAX_REF_FRAMES];
   int no0 = 0, no1 = 0;

   for (int i = 0; i < nst; i++)
      if (st[i].frame_poc < cur_field_poc) o0[no0++] = st[i];
   for (int i = 1; i < no0; i++) {
      struct anv_h264_fref v = o0[i]; int j = i - 1;
      while (j >= 0 && o0[j].frame_poc < v.frame_poc) { o0[j+1] = o0[j]; j--; }
      o0[j+1] = v;
   }
   int split0 = no0;
   for (int i = 0; i < nst; i++)
      if (st[i].frame_poc >= cur_field_poc) o0[no0++] = st[i];
   for (int i = split0 + 1; i < no0; i++) {
      struct anv_h264_fref v = o0[i]; int j = i - 1;
      while (j >= split0 && o0[j].frame_poc > v.frame_poc) { o0[j+1] = o0[j]; j--; }
      o0[j+1] = v;
   }

   for (int i = 0; i < nst; i++)
      if (st[i].frame_poc >= cur_field_poc) o1[no1++] = st[i];
   for (int i = 1; i < no1; i++) {
      struct anv_h264_fref v = o1[i]; int j = i - 1;
      while (j >= 0 && o1[j].frame_poc > v.frame_poc) { o1[j+1] = o1[j]; j--; }
      o1[j+1] = v;
   }
   int split1 = no1;
   for (int i = 0; i < nst; i++)
      if (st[i].frame_poc < cur_field_poc) o1[no1++] = st[i];
   for (int i = split1 + 1; i < no1; i++) {
      struct anv_h264_fref v = o1[i]; int j = i - 1;
      while (j >= split1 && o1[j].frame_poc < v.frame_poc) { o1[j+1] = o1[j]; j--; }
      o1[j+1] = v;
   }

   int idx0 = 0, idx1 = 0;
   anv_h264_emit_field_list(l0, l0b, ANV_H264_MAX_REF_FRAMES, &idx0, o0, no0, sel);
   anv_h264_emit_field_list(l0, l0b, ANV_H264_MAX_REF_FRAMES, &idx0, lt, nlt, sel);
   anv_h264_emit_field_list(l1, l1b, ANV_H264_MAX_REF_FRAMES, &idx1, o1, no1, sel);
   anv_h264_emit_field_list(l1, l1b, ANV_H264_MAX_REF_FRAMES, &idx1, lt, nlt, sel);
}

static void
anv_h264_build_default_ref_list(
   int slice_type,
   int32_t curr_poc,
   uint32_t ref_slot_count,
   const VkVideoReferenceSlotInfoKHR *ref_slots,
   uint8_t list0[ANV_H264_MAX_REF_FRAMES],
   uint8_t list1[ANV_H264_MAX_REF_FRAMES])
{
   memset(list0, ANV_H264_INVALID_SLOT, ANV_H264_MAX_REF_FRAMES);
   memset(list1, ANV_H264_INVALID_SLOT, ANV_H264_MAX_REF_FRAMES);

   if (slice_type == ANV_H264_SLICE_I || slice_type == ANV_H264_SLICE_SI)
      return;

   uint8_t st_slots[ANV_H264_MAX_REF_FRAMES];
   int32_t st_poc[ANV_H264_MAX_REF_FRAMES];
   int st_count = 0;

   uint8_t lt_slots[ANV_H264_MAX_REF_FRAMES];
   uint32_t lt_idx[ANV_H264_MAX_REF_FRAMES];
   int lt_count = 0;

   for (uint32_t i = 0; i < ref_slot_count; i++) {
      const VkVideoDecodeH264DpbSlotInfoKHR *dpb =
         vk_find_struct_const(ref_slots[i].pNext,
                              VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR);
      if (!dpb) continue;
      const StdVideoDecodeH264ReferenceInfo *ri = dpb->pStdReferenceInfo;
      if (!ri) continue;
      int slot = ref_slots[i].slotIndex;
      if (ri->flags.used_for_long_term_reference) {
         lt_slots[lt_count] = (uint8_t)slot;
         lt_idx[lt_count] = ri->FrameNum;
         lt_count++;
      } else {
         st_slots[st_count] = (uint8_t)slot;
         st_poc[st_count] = ri->PicOrderCnt[0];
         st_count++;
      }
   }

   for (int i = 1; i < st_count; i++) {
      uint8_t si = st_slots[i]; int32_t pi = st_poc[i];
      int j = i - 1;
      while (j >= 0 && st_poc[j] < pi) {
         st_slots[j+1] = st_slots[j]; st_poc[j+1] = st_poc[j]; j--;
      }
      st_slots[j+1] = si; st_poc[j+1] = pi;
   }

   for (int i = 1; i < lt_count; i++) {
      uint8_t si = lt_slots[i]; uint32_t li = lt_idx[i];
      int j = i - 1;
      while (j >= 0 && lt_idx[j] > li) {
         lt_slots[j+1] = lt_slots[j]; lt_idx[j+1] = lt_idx[j]; j--;
      }
      lt_slots[j+1] = si; lt_idx[j+1] = li;
   }

   if (slice_type == ANV_H264_SLICE_P || slice_type == ANV_H264_SLICE_SP) {
      int p = 0;
      for (int i = 0; i < st_count && p < ANV_H264_MAX_REF_FRAMES; i++)
         list0[p++] = st_slots[i];
      for (int i = 0; i < lt_count && p < ANV_H264_MAX_REF_FRAMES; i++)
         list0[p++] = lt_slots[i];
      return;
   }

   uint8_t past_slots[ANV_H264_MAX_REF_FRAMES];
   int32_t past_poc[ANV_H264_MAX_REF_FRAMES];
   int past_cnt = 0;
   uint8_t fut_slots[ANV_H264_MAX_REF_FRAMES];
   int32_t fut_poc[ANV_H264_MAX_REF_FRAMES];
   int fut_cnt = 0;

   for (int i = 0; i < st_count; i++) {
      if (st_poc[i] < curr_poc) {
         past_slots[past_cnt] = st_slots[i];
         past_poc[past_cnt++] = st_poc[i];
      }
   }
   for (int i = st_count - 1; i >= 0; i--) {
      if (st_poc[i] >= curr_poc) {
         fut_slots[fut_cnt] = st_slots[i];
         fut_poc[fut_cnt++] = st_poc[i];
      }
   }
   (void)past_poc; (void)fut_poc;

   int p0 = 0, p1 = 0;
   for (int i = 0; i < past_cnt && p0 < ANV_H264_MAX_REF_FRAMES; i++)
      list0[p0++] = past_slots[i];
   for (int i = 0; i < fut_cnt  && p0 < ANV_H264_MAX_REF_FRAMES; i++)
      list0[p0++] = fut_slots[i];
   for (int i = 0; i < lt_count && p0 < ANV_H264_MAX_REF_FRAMES; i++)
      list0[p0++] = lt_slots[i];

   for (int i = 0; i < fut_cnt  && p1 < ANV_H264_MAX_REF_FRAMES; i++)
      list1[p1++] = fut_slots[i];
   for (int i = 0; i < past_cnt && p1 < ANV_H264_MAX_REF_FRAMES; i++)
      list1[p1++] = past_slots[i];
   for (int i = 0; i < lt_count && p1 < ANV_H264_MAX_REF_FRAMES; i++)
      list1[p1++] = lt_slots[i];
}

static void
anv_h264_apply_ref_pic_list_mod(
   struct anv_h264_rbsp *st,
   uint32_t ref_slot_count,
   const VkVideoReferenceSlotInfoKHR *ref_slots,
   uint32_t curr_frame_num,
   uint32_t MaxFrameNum,
   uint8_t list[ANV_H264_MAX_REF_FRAMES],
   int active_count)
{
   int mod_flag = (int)anv_h264_rbsp_u(st, 1);
   if (!mod_flag || st->error) return;

   uint32_t picNumL = curr_frame_num;
   int refIdxL = 0;

   unsigned idc;
   do {
      idc = anv_h264_rbsp_ue(st);
      if (st->error) return;
      if (idc == 3) break;
      if (idc > 3) { st->error = true; return; }

      if (idc == 0 || idc == 1) {
         unsigned diff = anv_h264_rbsp_ue(st) + 1;
         if (st->error) return;
         if (idc == 0)
            picNumL = (picNumL - diff + MaxFrameNum) % MaxFrameNum;
         else
            picNumL = (picNumL + diff) % MaxFrameNum;

         uint8_t found_slot = ANV_H264_INVALID_SLOT;
         for (uint32_t j = 0; j < ref_slot_count; j++) {
            const VkVideoDecodeH264DpbSlotInfoKHR *dpb =
               vk_find_struct_const(ref_slots[j].pNext,
                                    VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR);
            if (!dpb || !dpb->pStdReferenceInfo) continue;
            if (!dpb->pStdReferenceInfo->flags.used_for_long_term_reference &&
                dpb->pStdReferenceInfo->FrameNum == picNumL) {
               found_slot = (uint8_t)ref_slots[j].slotIndex;
               break;
            }
         }
         if (found_slot == ANV_H264_INVALID_SLOT) continue;

         for (int k = active_count - 1; k > refIdxL; k--)
            list[k] = list[k - 1];
         list[refIdxL] = found_slot;

         int w = refIdxL + 1;
         for (int k = refIdxL + 1; k < active_count + 1 && w < ANV_H264_MAX_REF_FRAMES; k++) {
            if (list[k] != found_slot)
               list[w++] = list[k];
         }
         if (w < ANV_H264_MAX_REF_FRAMES)
            list[w] = ANV_H264_INVALID_SLOT;

         refIdxL++;
      } else if (idc == 2) {
         anv_h264_rbsp_ue(st);
         if (st->error) return;
      }
   } while (idc != 3);
}

/* H.264 8.2.4.3.1 */
static void
anv_h264_apply_ref_pic_list_mod_field(
   struct anv_h264_rbsp *st_rbsp,
   uint32_t ref_slot_count,
   const VkVideoReferenceSlotInfoKHR *ref_slots,
   uint32_t cur_frame_num,
   uint32_t MaxFrameNum,
   int sel,
   uint8_t list[ANV_H264_MAX_REF_FRAMES],
   uint8_t listb[ANV_H264_MAX_REF_FRAMES],
   int active_count)
{
   int mod_flag = (int)anv_h264_rbsp_u(st_rbsp, 1);
   if (!mod_flag || st_rbsp->error) return;

   struct anv_h264_fref st[ANV_H264_MAX_REF_FRAMES];
   struct anv_h264_fref lt[ANV_H264_MAX_REF_FRAMES];
   int nst = 0, nlt = 0;
   anv_h264_gather_frefs(ref_slot_count, ref_slots, cur_frame_num, MaxFrameNum,
                         st, &nst, lt, &nlt);

   int32_t CurrPicNum = 2 * (int32_t)cur_frame_num + 1;
   int32_t MaxPicNum = 2 * (int32_t)MaxFrameNum;
   int32_t picNumPred = CurrPicNum;
   int refIdx = 0;

   unsigned idc;
   do {
      idc = anv_h264_rbsp_ue(st_rbsp);
      if (st_rbsp->error) return;
      if (idc == 3) break;
      if (idc > 3) { st_rbsp->error = true; return; }

      uint8_t found_slot = ANV_H264_INVALID_SLOT;
      uint8_t found_bottom = 0;

      if (idc == 0 || idc == 1) {
         int32_t diff = (int32_t)anv_h264_rbsp_ue(st_rbsp) + 1;
         if (st_rbsp->error) return;
         int32_t noWrap;
         if (idc == 0)
            noWrap = (picNumPred - diff < 0) ?
                     picNumPred - diff + MaxPicNum : picNumPred - diff;
         else
            noWrap = (picNumPred + diff >= MaxPicNum) ?
                     picNumPred + diff - MaxPicNum : picNumPred + diff;
         picNumPred = noWrap;
         int32_t picNum = (noWrap > CurrPicNum) ? noWrap - MaxPicNum : noWrap;

         int32_t want_fnw = picNum >> 1;
         int want_par = (picNum & 1) ? sel : (sel ^ 3);
         for (int i = 0; i < nst; i++) {
            if (st[i].fnwrap == want_fnw && (st[i].parities & want_par)) {
               found_slot = st[i].slot;
               found_bottom = (want_par == ANV_H264_PAR_BOTTOM);
               break;
            }
         }
      } else {
         uint32_t ltpn = anv_h264_rbsp_ue(st_rbsp);
         if (st_rbsp->error) return;
         uint32_t want_lt = ltpn >> 1;
         int want_par = (ltpn & 1) ? sel : (sel ^ 3);
         for (int i = 0; i < nlt; i++) {
            if (lt[i].lt_idx == want_lt && (lt[i].parities & want_par)) {
               found_slot = lt[i].slot;
               found_bottom = (want_par == ANV_H264_PAR_BOTTOM);
               break;
            }
         }
      }

      if (found_slot == ANV_H264_INVALID_SLOT) continue;

      for (int k = active_count - 1; k > refIdx && k < ANV_H264_MAX_REF_FRAMES; k--) {
         list[k] = list[k - 1];
         listb[k] = listb[k - 1];
      }
      if (refIdx < ANV_H264_MAX_REF_FRAMES) {
         list[refIdx] = found_slot;
         listb[refIdx] = found_bottom;
      }

      int w = refIdx + 1;
      for (int k = refIdx + 1;
           k < active_count + 1 && k < ANV_H264_MAX_REF_FRAMES &&
           w < ANV_H264_MAX_REF_FRAMES; k++) {
         if (!(list[k] == found_slot && listb[k] == found_bottom)) {
            list[w] = list[k];
            listb[w] = listb[k];
            w++;
         }
      }
      if (w < ANV_H264_MAX_REF_FRAMES)
         list[w] = ANV_H264_INVALID_SLOT;

      refIdx++;
   } while (idc != 3);
}

static void
anv_h264_parse_pred_weight_table(
   struct anv_h264_rbsp *st,
   int ChromaArrayType,
   int num_l0,
   int num_l1,
   int slice_type,
   struct anv_h264_slice_params *out)
{
   out->luma_log2_weight_denom = (int)anv_h264_rbsp_ue(st);
   out->chroma_log2_weight_denom = 0;
   if (ChromaArrayType != 0)
      out->chroma_log2_weight_denom = (int)anv_h264_rbsp_ue(st);
   if (st->error) return;

   if (out->luma_log2_weight_denom > 7 ||
       (ChromaArrayType != 0 && out->chroma_log2_weight_denom > 7)) {
      st->error = true;
      return;
   }

   int def_luma = 1 << out->luma_log2_weight_denom;
   int def_chroma = 1 << out->chroma_log2_weight_denom;

   for (int i = 0; i < num_l0; i++) {
      out->luma_weight_l0[i] = def_luma; out->luma_offset_l0[i] = 0;
      for (int c = 0; c < 2; c++) {
         out->chroma_weight_l0[i][c] = def_chroma;
         out->chroma_offset_l0[i][c] = 0;
      }
   }
   for (int i = 0; i < num_l1; i++) {
      out->luma_weight_l1[i] = def_luma; out->luma_offset_l1[i] = 0;
      for (int c = 0; c < 2; c++) {
         out->chroma_weight_l1[i][c] = def_chroma;
         out->chroma_offset_l1[i][c] = 0;
      }
   }

   for (int i = 0; i < num_l0 && !st->error; i++) {
      if (anv_h264_rbsp_u(st, 1)) {
         out->luma_weight_l0[i] = (int16_t)anv_h264_rbsp_se(st);
         out->luma_offset_l0[i] = (int16_t)anv_h264_rbsp_se(st);
         if (out->luma_weight_l0[i] != def_luma || out->luma_offset_l0[i])
            out->has_weight_offsets_l0 = true;
      }
      if (ChromaArrayType != 0 && anv_h264_rbsp_u(st, 1)) {
         for (int c = 0; c < 2; c++) {
            out->chroma_weight_l0[i][c] = (int16_t)anv_h264_rbsp_se(st);
            out->chroma_offset_l0[i][c] = (int16_t)anv_h264_rbsp_se(st);
            if (out->chroma_weight_l0[i][c] != def_chroma ||
                out->chroma_offset_l0[i][c])
               out->has_weight_offsets_l0 = true;
         }
      }
   }

   if (slice_type != ANV_H264_SLICE_B) return;

   for (int i = 0; i < num_l1 && !st->error; i++) {
      if (anv_h264_rbsp_u(st, 1)) {
         out->luma_weight_l1[i] = (int16_t)anv_h264_rbsp_se(st);
         out->luma_offset_l1[i] = (int16_t)anv_h264_rbsp_se(st);
         if (out->luma_weight_l1[i] != def_luma || out->luma_offset_l1[i])
            out->has_weight_offsets_l1 = true;
      }
      if (ChromaArrayType != 0 && anv_h264_rbsp_u(st, 1)) {
         for (int c = 0; c < 2; c++) {
            out->chroma_weight_l1[i][c] = (int16_t)anv_h264_rbsp_se(st);
            out->chroma_offset_l1[i][c] = (int16_t)anv_h264_rbsp_se(st);
            if (out->chroma_weight_l1[i][c] != def_chroma ||
                out->chroma_offset_l1[i][c])
               out->has_weight_offsets_l1 = true;
         }
      }
   }
}

static void
anv_h264_consume_dec_ref_pic_marking(struct anv_h264_rbsp *st, bool is_idr)
{
   if (is_idr) {
      anv_h264_rbsp_u(st, 1);
      anv_h264_rbsp_u(st, 1);
   } else {
      if (anv_h264_rbsp_u(st, 1)) {
         unsigned op;
         do {
            op = anv_h264_rbsp_ue(st);
            if (st->error) return;
            if (op > 6) { st->error = true; return; }
            if (op == 1 || op == 3) anv_h264_rbsp_ue(st);
            if (op == 2)            anv_h264_rbsp_ue(st);
            if (op == 3 || op == 6) anv_h264_rbsp_ue(st);
            if (op == 4)            anv_h264_rbsp_ue(st);
         } while (op != 0 && !st->error);
      }
   }
}

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
   struct anv_h264_slice_params *out)
{
   memset(out, 0, sizeof(*out));
   out->slice_type = ANV_H264_SLICE_I;

   for (int i = 0; i < ANV_H264_MAX_REF_FRAMES; i++) {
      out->luma_weight_l0[i] = 1; out->luma_weight_l1[i] = 1;
      for (int c = 0; c < 2; c++) {
         out->chroma_weight_l0[i][c] = 1;
         out->chroma_weight_l1[i][c] = 1;
      }
   }

   memset(out->ref_list0, ANV_H264_INVALID_SLOT, sizeof(out->ref_list0));
   memset(out->ref_list1, ANV_H264_INVALID_SLOT, sizeof(out->ref_list1));

   struct anv_h264_rbsp st;
   anv_h264_rbsp_init(&st, buf + slice_nalu_offset, slice_nalu_size);

   anv_h264_rbsp_u(&st, 1);
   int nal_ref_idc = (int)anv_h264_rbsp_u(&st, 2);
   int nal_unit_type = (int)anv_h264_rbsp_u(&st, 5);

   VALID_RBSP_OR_BAIL(st)

   bool is_idr = (nal_unit_type == 5);

   out->first_mb_in_slice = (int)anv_h264_rbsp_ue(&st);
   int raw_slice_type = (int)anv_h264_rbsp_ue(&st);

   VALID_RBSP_OR_BAIL(st)

   int slice_type = raw_slice_type;
   if (slice_type > 4) slice_type -= 5;
   out->slice_type = slice_type;

   bool cur_field_pic = pic_info->pStdPictureInfo->flags.field_pic_flag;
   bool cur_bottom = pic_info->pStdPictureInfo->flags.bottom_field_flag;
   int cur_sel = (cur_field_pic && cur_bottom) ?
                 ANV_H264_PAR_BOTTOM : ANV_H264_PAR_TOP;
   uint32_t cur_frame_num = pic_info->pStdPictureInfo->frame_num;
   uint32_t MaxFrameNumForList = 1u << (sps->log2_max_frame_num_minus4 + 4);

   int32_t curr_poc = cur_field_pic && cur_bottom ?
                      pic_info->pStdPictureInfo->PicOrderCnt[1] :
                      pic_info->pStdPictureInfo->PicOrderCnt[0];

   if (cur_field_pic) {
      anv_h264_build_field_ref_lists(slice_type, cur_sel, curr_poc,
                                     cur_frame_num, MaxFrameNumForList,
                                     ref_slot_count, ref_slots,
                                     out->ref_list0, out->ref_list0_bottom,
                                     out->ref_list1, out->ref_list1_bottom);
   } else {
      anv_h264_build_default_ref_list(slice_type, curr_poc,
                                      ref_slot_count, ref_slots,
                                      out->ref_list0, out->ref_list1);
   }

   anv_h264_rbsp_ue(&st);

   if (sps->flags.separate_colour_plane_flag)
      anv_h264_rbsp_u(&st, 2);

   uint32_t log2_max_fn = sps->log2_max_frame_num_minus4 + 4;
   uint32_t frame_num = anv_h264_rbsp_u(&st, log2_max_fn);
   uint32_t MaxFrameNum = 1u << log2_max_fn;

   bool field_pic_flag = false;
   if (!sps->flags.frame_mbs_only_flag) {
      field_pic_flag = !!anv_h264_rbsp_u(&st, 1);
      if (field_pic_flag) anv_h264_rbsp_u(&st, 1);
   }
   (void)field_pic_flag;

   if (is_idr) anv_h264_rbsp_ue(&st);

   if (sps->pic_order_cnt_type == 0) {
      anv_h264_rbsp_u(&st, sps->log2_max_pic_order_cnt_lsb_minus4 + 4);
      if (pps->flags.bottom_field_pic_order_in_frame_present_flag &&
          !pic_info->pStdPictureInfo->flags.field_pic_flag)
         anv_h264_rbsp_se(&st);
   } else if (sps->pic_order_cnt_type == 1 &&
              !sps->flags.delta_pic_order_always_zero_flag) {
      anv_h264_rbsp_se(&st);
      if (pps->flags.bottom_field_pic_order_in_frame_present_flag &&
          !pic_info->pStdPictureInfo->flags.field_pic_flag)
         anv_h264_rbsp_se(&st);
   }

   if (pps->flags.redundant_pic_cnt_present_flag)
      anv_h264_rbsp_ue(&st);

   if (slice_type == ANV_H264_SLICE_B)
      out->direct_spatial_mv_pred_flag = (int)anv_h264_rbsp_u(&st, 1);

   out->num_ref_idx_l0_active_minus1 = pps->num_ref_idx_l0_default_active_minus1;
   out->num_ref_idx_l1_active_minus1 = pps->num_ref_idx_l1_default_active_minus1;
   if (slice_type == ANV_H264_SLICE_P || slice_type == ANV_H264_SLICE_SP ||
       slice_type == ANV_H264_SLICE_B) {
      if (anv_h264_rbsp_u(&st, 1)) {
         out->num_ref_idx_l0_active_minus1 = (int)anv_h264_rbsp_ue(&st);
         if (slice_type == ANV_H264_SLICE_B)
            out->num_ref_idx_l1_active_minus1 = (int)anv_h264_rbsp_ue(&st);
      }
   }

   VALID_RBSP_OR_BAIL(st)

   int active_l0 = out->num_ref_idx_l0_active_minus1 + 1;
   int active_l1 = out->num_ref_idx_l1_active_minus1 + 1;
   if (active_l0 > ANV_H264_MAX_REF_FRAMES) active_l0 = ANV_H264_MAX_REF_FRAMES;
   if (active_l1 > ANV_H264_MAX_REF_FRAMES) active_l1 = ANV_H264_MAX_REF_FRAMES;

   if (slice_type != ANV_H264_SLICE_I && slice_type != ANV_H264_SLICE_SI) {
      if (cur_field_pic) {
         anv_h264_apply_ref_pic_list_mod_field(&st, ref_slot_count, ref_slots,
                                               frame_num, MaxFrameNum, cur_sel,
                                               out->ref_list0,
                                               out->ref_list0_bottom, active_l0);
         if (slice_type == ANV_H264_SLICE_B)
            anv_h264_apply_ref_pic_list_mod_field(&st, ref_slot_count, ref_slots,
                                                  frame_num, MaxFrameNum, cur_sel,
                                                  out->ref_list1,
                                                  out->ref_list1_bottom, active_l1);
      } else {
         anv_h264_apply_ref_pic_list_mod(&st, ref_slot_count, ref_slots,
                                         frame_num, MaxFrameNum,
                                         out->ref_list0, active_l0);
         if (slice_type == ANV_H264_SLICE_B)
            anv_h264_apply_ref_pic_list_mod(&st, ref_slot_count, ref_slots,
                                            frame_num, MaxFrameNum,
                                            out->ref_list1, active_l1);
      }
   }

   VALID_RBSP_OR_BAIL(st)

   bool need_pwt_l0 = (pps->flags.weighted_pred_flag &&
                       (slice_type == ANV_H264_SLICE_P ||
                        slice_type == ANV_H264_SLICE_SP));
   bool need_pwt_l1 = (pps->weighted_bipred_idc == 1 &&
                       slice_type == ANV_H264_SLICE_B);
   if (need_pwt_l0 || need_pwt_l1) {
      anv_h264_parse_pred_weight_table(&st, sps->chroma_format_idc,
                                       active_l0, active_l1,
                                       slice_type, out);
   }

   if (nal_ref_idc != 0)
      anv_h264_consume_dec_ref_pic_marking(&st, is_idr);

   VALID_RBSP_OR_BAIL(st)

   out->cabac_init_idc = 0;
   if (pps->flags.entropy_coding_mode_flag &&
       slice_type != ANV_H264_SLICE_I &&
       slice_type != ANV_H264_SLICE_SI)
      out->cabac_init_idc = (int)anv_h264_rbsp_ue(&st);

   out->slice_qp_delta = anv_h264_rbsp_se(&st);

   if (slice_type == ANV_H264_SLICE_SP || slice_type == ANV_H264_SLICE_SI) {
      if (slice_type == ANV_H264_SLICE_SP)
         anv_h264_rbsp_u(&st, 1);
      anv_h264_rbsp_se(&st);
   }

   if (pps->flags.deblocking_filter_control_present_flag) {
      out->disable_deblocking_filter_idc = (int)anv_h264_rbsp_ue(&st);
      if (out->disable_deblocking_filter_idc != 1) {
         out->slice_alpha_c0_offset_div2 = anv_h264_rbsp_se(&st);
         out->slice_beta_offset_div2 = anv_h264_rbsp_se(&st);
      }
   }

   /* Let the caller fall back to concealment if the parser lost bit-alignment. */
   if (out->slice_alpha_c0_offset_div2 < -6 || out->slice_alpha_c0_offset_div2 > 6 ||
       out->slice_beta_offset_div2 < -6 || out->slice_beta_offset_div2 > 6 ||
       out->slice_qp_delta < -51 || out->slice_qp_delta > 51 ||
       out->disable_deblocking_filter_idc > 2 ||
       out->cabac_init_idc > 2)
      st.error = true;

   VALID_RBSP_OR_BAIL(st)

   if (pps->flags.entropy_coding_mode_flag) {
      while ((st.bits_consumed & 7) && !st.error)
         anv_h264_rbsp_consume_bit(&st);
   }
   out->first_mb_byte_offset = (uint32_t)(st.bits_consumed >> 3);
   out->first_mb_bit_offset = (uint32_t)(st.bits_consumed & 7);

   return !st.error;
}
