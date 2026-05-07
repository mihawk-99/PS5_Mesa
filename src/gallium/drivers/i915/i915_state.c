/**************************************************************************
 *
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/* Authors:  Keith Whitwell <keithw@vmware.com>
 */

#include "compiler/nir/nir_builder.h"
#include "draw/draw_context.h"
#include "nir/nir_to_tgsi.h"
#include "tgsi/tgsi_from_mesa.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_transfer.h"
#include "nir.h"

#include "i915_context.h"
#include "i915_debug.h"
#include "i915_fpc.h"
#include "i915_reg.h"
#include "i915_resource.h"
#include "i915_state.h"
#include "i915_state_inlines.h"
#include "i915_surface.h"

static void i915_delete_fs_state(struct pipe_context *pipe, void *shader);

/* The i915 (and related graphics cores) do not support GL_CLAMP.  The
 * Intel drivers for "other operating systems" implement GL_CLAMP as
 * GL_CLAMP_TO_EDGE, so the same is done here.
 */
static unsigned
translate_wrap_mode(unsigned wrap)
{
   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT:
      return TEXCOORDMODE_WRAP;
   case PIPE_TEX_WRAP_CLAMP:
      return TEXCOORDMODE_CLAMP_EDGE; /* not quite correct */
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      return TEXCOORDMODE_CLAMP_EDGE;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      return TEXCOORDMODE_CLAMP_BORDER;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      return TEXCOORDMODE_MIRROR;
   default:
      return TEXCOORDMODE_WRAP;
   }
}

static unsigned
translate_img_filter(unsigned filter)
{
   switch (filter) {
   case PIPE_TEX_FILTER_NEAREST:
      return FILTER_NEAREST;
   case PIPE_TEX_FILTER_LINEAR:
      return FILTER_LINEAR;
   default:
      assert(0);
      return FILTER_NEAREST;
   }
}

static unsigned
translate_mip_filter(unsigned filter)
{
   switch (filter) {
   case PIPE_TEX_MIPFILTER_NONE:
      return MIPFILTER_NONE;
   case PIPE_TEX_MIPFILTER_NEAREST:
      return MIPFILTER_NEAREST;
   case PIPE_TEX_MIPFILTER_LINEAR:
      return MIPFILTER_LINEAR;
   default:
      assert(0);
      return MIPFILTER_NONE;
   }
}

static uint32_t
i915_remap_lis6_blend_dst_alpha(uint32_t lis6, uint32_t normal, uint32_t inv)
{
   uint32_t src = (lis6 >> S6_CBUF_SRC_BLEND_FACT_SHIFT) & BLENDFACT_MASK;
   lis6 &= ~SRC_BLND_FACT(BLENDFACT_MASK);
   if (src == BLENDFACT_DST_ALPHA)
      src = normal;
   else if (src == BLENDFACT_INV_DST_ALPHA)
      src = inv;
   lis6 |= SRC_BLND_FACT(src);

   uint32_t dst = (lis6 >> S6_CBUF_DST_BLEND_FACT_SHIFT) & BLENDFACT_MASK;
   lis6 &= ~DST_BLND_FACT(BLENDFACT_MASK);
   if (dst == BLENDFACT_DST_ALPHA)
      dst = normal;
   else if (dst == BLENDFACT_INV_DST_ALPHA)
      dst = inv;
   lis6 |= DST_BLND_FACT(dst);

   return lis6;
}

static uint32_t
i915_remap_iab_blend_dst_alpha(uint32_t iab, uint32_t normal, uint32_t inv)
{
   uint32_t src = (iab >> IAB_SRC_FACTOR_SHIFT) & BLENDFACT_MASK;
   iab &= ~SRC_BLND_FACT(BLENDFACT_MASK);
   if (src == BLENDFACT_DST_ALPHA)
      src = normal;
   else if (src == BLENDFACT_INV_DST_ALPHA)
      src = inv;
   iab |= SRC_ABLND_FACT(src);

   uint32_t dst = (iab >> IAB_DST_FACTOR_SHIFT) & BLENDFACT_MASK;
   iab &= ~DST_BLND_FACT(BLENDFACT_MASK);
   if (dst == BLENDFACT_DST_ALPHA)
      dst = normal;
   else if (dst == BLENDFACT_INV_DST_ALPHA)
      dst = inv;
   iab |= DST_ABLND_FACT(dst);

   return iab;
}

/* None of this state is actually used for anything yet.
 */
static void *
i915_create_blend_state(struct pipe_context *pipe,
                        const struct pipe_blend_state *blend)
{
   struct i915_blend_state *cso_data = CALLOC_STRUCT(i915_blend_state);

   {
      unsigned eqRGB = blend->rt[0].rgb_func;
      unsigned srcRGB = blend->rt[0].rgb_src_factor;
      unsigned dstRGB = blend->rt[0].rgb_dst_factor;

      unsigned eqA = blend->rt[0].alpha_func;
      unsigned srcA = blend->rt[0].alpha_src_factor;
      unsigned dstA = blend->rt[0].alpha_dst_factor;

      /* Special handling for MIN/MAX filter modes handled at
       * frontend level.
       */

      if (srcA != srcRGB || dstA != dstRGB || eqA != eqRGB) {

         cso_data->iab = (_3DSTATE_INDEPENDENT_ALPHA_BLEND_CMD |
                          IAB_MODIFY_ENABLE | IAB_ENABLE | IAB_MODIFY_FUNC |
                          IAB_MODIFY_SRC_FACTOR | IAB_MODIFY_DST_FACTOR |
                          SRC_ABLND_FACT(i915_translate_blend_factor(srcA)) |
                          DST_ABLND_FACT(i915_translate_blend_factor(dstA)) |
                          (i915_translate_blend_func(eqA) << IAB_FUNC_SHIFT));
      } else {
         cso_data->iab =
            (_3DSTATE_INDEPENDENT_ALPHA_BLEND_CMD | IAB_MODIFY_ENABLE | 0);
      }
   }

   cso_data->modes4 |=
      (_3DSTATE_MODES_4_CMD | ENABLE_LOGIC_OP_FUNC |
       LOGIC_OP_FUNC(i915_translate_logic_op(blend->logicop_func)));

   if (blend->logicop_enable)
      cso_data->LIS5 |= S5_LOGICOP_ENABLE;

   if (blend->dither)
      cso_data->LIS5 |= S5_COLOR_DITHER_ENABLE;

   /* We potentially do some fixup at emission for non-BGRA targets */
   if ((blend->rt[0].colormask & PIPE_MASK_R) == 0)
      cso_data->LIS5 |= S5_WRITEDISABLE_RED;

   if ((blend->rt[0].colormask & PIPE_MASK_G) == 0)
      cso_data->LIS5 |= S5_WRITEDISABLE_GREEN;

   if ((blend->rt[0].colormask & PIPE_MASK_B) == 0)
      cso_data->LIS5 |= S5_WRITEDISABLE_BLUE;

   if ((blend->rt[0].colormask & PIPE_MASK_A) == 0)
      cso_data->LIS5 |= S5_WRITEDISABLE_ALPHA;

   if (blend->rt[0].blend_enable) {
      unsigned funcRGB = blend->rt[0].rgb_func;
      unsigned srcRGB = blend->rt[0].rgb_src_factor;
      unsigned dstRGB = blend->rt[0].rgb_dst_factor;

      cso_data->LIS6 |=
         (S6_CBUF_BLEND_ENABLE |
          SRC_BLND_FACT(i915_translate_blend_factor(srcRGB)) |
          DST_BLND_FACT(i915_translate_blend_factor(dstRGB)) |
          (i915_translate_blend_func(funcRGB) << S6_CBUF_BLEND_FUNC_SHIFT));
   }

   cso_data->LIS6_alpha_in_g = i915_remap_lis6_blend_dst_alpha(
      cso_data->LIS6, BLENDFACT_DST_COLR, BLENDFACT_INV_DST_COLR);
   cso_data->LIS6_alpha_is_x = i915_remap_lis6_blend_dst_alpha(
      cso_data->LIS6, BLENDFACT_ONE, BLENDFACT_ZERO);

   cso_data->iab_alpha_in_g = i915_remap_iab_blend_dst_alpha(
      cso_data->iab, BLENDFACT_DST_COLR, BLENDFACT_INV_DST_COLR);
   cso_data->iab_alpha_is_x = i915_remap_iab_blend_dst_alpha(
      cso_data->iab, BLENDFACT_ONE, BLENDFACT_ZERO);

   return cso_data;
}

static void
i915_bind_blend_state(struct pipe_context *pipe, void *blend)
{
   struct i915_context *i915 = i915_context(pipe);

   if (i915->blend == blend)
      return;

   i915->blend = (struct i915_blend_state *)blend;

   i915->dirty |= I915_NEW_BLEND;
}

static void
i915_delete_blend_state(struct pipe_context *pipe, void *blend)
{
   FREE(blend);
}

static void
i915_set_blend_color(struct pipe_context *pipe,
                     const struct pipe_blend_color *blend_color)
{
   struct i915_context *i915 = i915_context(pipe);

   if (!blend_color)
      return;

   i915->blend_color = *blend_color;

   i915->dirty |= I915_NEW_BLEND;
}

static void
i915_set_stencil_ref(struct pipe_context *pipe,
                     const struct pipe_stencil_ref stencil_ref)
{
   struct i915_context *i915 = i915_context(pipe);

   i915->stencil_ref = stencil_ref;

   i915->dirty |= I915_NEW_DEPTH_STENCIL;
}

static void *
i915_create_sampler_state(struct pipe_context *pipe,
                          const struct pipe_sampler_state *sampler)
{
   struct i915_sampler_state *cso = CALLOC_STRUCT(i915_sampler_state);
   const unsigned ws = sampler->wrap_s;
   const unsigned wt = sampler->wrap_t;
   const unsigned wr = sampler->wrap_r;
   unsigned minFilt, magFilt;
   unsigned mipFilt;

   cso->templ = *sampler;

   mipFilt = translate_mip_filter(sampler->min_mip_filter);
   minFilt = translate_img_filter(sampler->min_img_filter);
   magFilt = translate_img_filter(sampler->mag_img_filter);

   if (sampler->max_anisotropy > 1)
      minFilt = magFilt = FILTER_ANISOTROPIC;

   if (sampler->max_anisotropy > 2) {
      cso->state[0] |= SS2_MAX_ANISO_4;
   }

   {
      int b = (int)(sampler->lod_bias * 16.0);
      b = CLAMP(b, -256, 255);
      cso->state[0] |= ((b << SS2_LOD_BIAS_SHIFT) & SS2_LOD_BIAS_MASK);
   }

   /* Shadow:
    */
   if (sampler->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE) {
      cso->state[0] |= (SS2_SHADOW_ENABLE | i915_translate_shadow_compare_func(
                                               sampler->compare_func));

      minFilt = FILTER_4X4_FLAT;
      magFilt = FILTER_4X4_FLAT;
   }

   cso->state[0] |=
      ((minFilt << SS2_MIN_FILTER_SHIFT) | (mipFilt << SS2_MIP_FILTER_SHIFT) |
       (magFilt << SS2_MAG_FILTER_SHIFT));

   cso->state[1] |= ((translate_wrap_mode(ws) << SS3_TCX_ADDR_MODE_SHIFT) |
                     (translate_wrap_mode(wt) << SS3_TCY_ADDR_MODE_SHIFT) |
                     (translate_wrap_mode(wr) << SS3_TCZ_ADDR_MODE_SHIFT));

   if (!sampler->unnormalized_coords)
      cso->state[1] |= SS3_NORMALIZED_COORDS;

   {
      int minlod = (int)(16.0 * sampler->min_lod);
      int maxlod = (int)(16.0 * sampler->max_lod);
      minlod = CLAMP(minlod, 0, 16 * 11);
      maxlod = CLAMP(maxlod, 0, 16 * 11);

      if (minlod > maxlod)
         maxlod = minlod;

      cso->minlod = minlod;
      cso->maxlod = maxlod;
   }

   {
      uint8_t r = float_to_ubyte(sampler->border_color.f[0]);
      uint8_t g = float_to_ubyte(sampler->border_color.f[1]);
      uint8_t b = float_to_ubyte(sampler->border_color.f[2]);
      uint8_t a = float_to_ubyte(sampler->border_color.f[3]);
      cso->state[2] = I915PACKCOLOR8888(r, g, b, a);
   }
   return cso;
}

static void
i915_bind_sampler_states(struct pipe_context *pipe,
                         mesa_shader_stage shader, unsigned start,
                         unsigned num, void **samplers)
{
   if (shader != MESA_SHADER_FRAGMENT) {
      assert(num == 0);
      return;
   }

   struct i915_context *i915 = i915_context(pipe);
   unsigned i;

   /* Check for no-op */
   if (num == i915->num_samplers &&
       !memcmp(i915->fragment_sampler + start, samplers, num * sizeof(void *)))
      return;

   for (i = 0; i < num; ++i)
      i915->fragment_sampler[i + start] = samplers[i];

   /* find highest non-null samplers[] entry */
   {
      unsigned j = MAX2(i915->num_samplers, start + num);
      while (j > 0 && i915->fragment_sampler[j - 1] == NULL)
         j--;
      i915->num_samplers = j;
   }

   i915->dirty |= I915_NEW_SAMPLER;
}

static void
i915_delete_sampler_state(struct pipe_context *pipe, void *sampler)
{
   FREE(sampler);
}

/** XXX move someday?  Or consolidate all these simple state setters
 * into one file.
 */

static uint32_t
i915_get_modes4_stencil(const struct pipe_stencil_state *stencil)
{
   int testmask = stencil->valuemask & 0xff;
   int writemask = stencil->writemask & 0xff;

   return (_3DSTATE_MODES_4_CMD | ENABLE_STENCIL_TEST_MASK |
           STENCIL_TEST_MASK(testmask) | ENABLE_STENCIL_WRITE_MASK |
           STENCIL_WRITE_MASK(writemask));
}

static uint32_t
i915_get_lis5_stencil(const struct pipe_stencil_state *stencil)
{
   int test = i915_translate_compare_func(stencil->func);
   int fop = i915_translate_stencil_op(stencil->fail_op);
   int dfop = i915_translate_stencil_op(stencil->zfail_op);
   int dpop = i915_translate_stencil_op(stencil->zpass_op);

   return (S5_STENCIL_TEST_ENABLE | S5_STENCIL_WRITE_ENABLE |
           (test << S5_STENCIL_TEST_FUNC_SHIFT) |
           (fop << S5_STENCIL_FAIL_SHIFT) |
           (dfop << S5_STENCIL_PASS_Z_FAIL_SHIFT) |
           (dpop << S5_STENCIL_PASS_Z_PASS_SHIFT));
}

static uint32_t
i915_get_bfo(const struct pipe_stencil_state *stencil)
{
   int test = i915_translate_compare_func(stencil->func);
   int fop = i915_translate_stencil_op(stencil->fail_op);
   int dfop = i915_translate_stencil_op(stencil->zfail_op);
   int dpop = i915_translate_stencil_op(stencil->zpass_op);

   return (_3DSTATE_BACKFACE_STENCIL_OPS | BFO_ENABLE_STENCIL_FUNCS |
           BFO_ENABLE_STENCIL_TWO_SIDE | BFO_ENABLE_STENCIL_REF |
           BFO_STENCIL_TWO_SIDE | (test << BFO_STENCIL_TEST_SHIFT) |
           (fop << BFO_STENCIL_FAIL_SHIFT) |
           (dfop << BFO_STENCIL_PASS_Z_FAIL_SHIFT) |
           (dpop << BFO_STENCIL_PASS_Z_PASS_SHIFT));
}

static uint32_t
i915_get_bfm(const struct pipe_stencil_state *stencil)
{
   return (_3DSTATE_BACKFACE_STENCIL_MASKS | BFM_ENABLE_STENCIL_TEST_MASK |
           BFM_ENABLE_STENCIL_WRITE_MASK |
           ((stencil->valuemask & 0xff) << BFM_STENCIL_TEST_MASK_SHIFT) |
           ((stencil->writemask & 0xff) << BFM_STENCIL_WRITE_MASK_SHIFT));
}

static void *
i915_create_depth_stencil_state(
   struct pipe_context *pipe,
   const struct pipe_depth_stencil_alpha_state *depth_stencil)
{
   struct i915_depth_stencil_state *cso =
      CALLOC_STRUCT(i915_depth_stencil_state);

   cso->stencil_modes4_cw = i915_get_modes4_stencil(&depth_stencil->stencil[0]);
   cso->stencil_modes4_ccw =
      i915_get_modes4_stencil(&depth_stencil->stencil[1]);

   if (depth_stencil->stencil[0].enabled) {
      cso->stencil_LIS5_cw = i915_get_lis5_stencil(&depth_stencil->stencil[0]);
   }

   if (depth_stencil->stencil[1].enabled) {
      cso->bfo_cw[0] = i915_get_bfo(&depth_stencil->stencil[1]);
      cso->bfo_cw[1] = i915_get_bfm(&depth_stencil->stencil[1]);

      /* Precompute the backface stencil settings if front winding order is
       * reversed -- HW doesn't have a bit to flip it for us.
       */
      cso->stencil_LIS5_ccw = i915_get_lis5_stencil(&depth_stencil->stencil[1]);
      cso->bfo_ccw[0] = i915_get_bfo(&depth_stencil->stencil[0]);
      cso->bfo_ccw[1] = i915_get_bfm(&depth_stencil->stencil[0]);
   } else {
      /* This actually disables two-side stencil: The bit set is a
       * modify-enable bit to indicate we are changing the two-side
       * setting.  Then there is a symbolic zero to show that we are
       * setting the flag to zero/off.
       */
      cso->bfo_cw[0] = cso->bfo_ccw[0] =
         (_3DSTATE_BACKFACE_STENCIL_OPS | BFO_ENABLE_STENCIL_TWO_SIDE | 0);
      cso->bfo_cw[1] = cso->bfo_ccw[1] = 0;

      cso->stencil_LIS5_ccw = cso->stencil_LIS5_cw;
   }

   if (depth_stencil->depth_enabled) {
      int func = i915_translate_compare_func(depth_stencil->depth_func);

      cso->depth_LIS6 |=
         (S6_DEPTH_TEST_ENABLE | (func << S6_DEPTH_TEST_FUNC_SHIFT));

      if (depth_stencil->depth_writemask)
         cso->depth_LIS6 |= S6_DEPTH_WRITE_ENABLE;
   }

   if (depth_stencil->alpha_enabled) {
      int test = i915_translate_compare_func(depth_stencil->alpha_func);
      uint8_t refByte = float_to_ubyte(depth_stencil->alpha_ref_value);

      cso->depth_LIS6 |=
         (S6_ALPHA_TEST_ENABLE | (test << S6_ALPHA_TEST_FUNC_SHIFT) |
          (((unsigned)refByte) << S6_ALPHA_REF_SHIFT));
   }

   return cso;
}

static void
i915_bind_depth_stencil_state(struct pipe_context *pipe, void *depth_stencil)
{
   struct i915_context *i915 = i915_context(pipe);

   if (i915->depth_stencil == depth_stencil)
      return;

   i915->depth_stencil = (const struct i915_depth_stencil_state *)depth_stencil;

   i915->dirty |= I915_NEW_DEPTH_STENCIL;
}

static void
i915_delete_depth_stencil_state(struct pipe_context *pipe, void *depth_stencil)
{
   FREE(depth_stencil);
}

static void
i915_set_scissor_states(struct pipe_context *pipe, unsigned start_slot,
                        unsigned num_scissors,
                        const struct pipe_scissor_state *scissor)
{
   struct i915_context *i915 = i915_context(pipe);

   memcpy(&i915->scissor, scissor, sizeof(*scissor));
   i915->dirty |= I915_NEW_SCISSOR;
}

static void
i915_set_polygon_stipple(struct pipe_context *pipe,
                         const struct pipe_poly_stipple *stipple)
{
}

static const struct nir_to_tgsi_options ntt_options = {
   .lower_fabs = true,
};

static int
type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static bool
scalarize_vector_bools(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_bcsel ||
          alu->op == nir_op_fcsel_ge ||
          alu->op == nir_op_fcsel_gt;
}

static bool
lower_fsqrt_filter(const nir_instr *instr, UNUSED const void *data)
{
   return instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == nir_op_fsqrt;
}

static nir_def *
lower_fsqrt_impl(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *src = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   return nir_fmul(b, src, nir_frsq(b, src));
}

static uint8_t
i915_vectorize_filter(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;

   switch (nir_instr_as_alu(instr)->op) {
   case nir_op_frcp:
   case nir_op_frsq:
   case nir_op_fsqrt:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_fpow:
      return 1;
   default:
      return 4;
   }
}

static char *
i915_check_control_flow(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_block *first = nir_start_block(impl);
   nir_cf_node *next = nir_cf_node_next(&first->cf_node);

   if (next) {
      switch (next->type) {
      case nir_cf_node_if:
         return "if/then statements not supported by i915 fragment shaders, "
                "should have been flattened by peephole_select.";
      case nir_cf_node_loop:
         return "looping not supported i915 fragment shaders, all loops "
                "must be statically unrollable.";
      default:
         return "Unknown control flow type";
      }
   }

   return NULL;
}

enum i915_fs_mode {
   I915_FS_TGSI,
   I915_FS_NIR,
   I915_FS_BOTH,
};

static enum i915_fs_mode
i915_get_fs_mode(void)
{
   const char *env = debug_get_option("I915_FS", "both");
   if (!strcmp(env, "tgsi"))
      return I915_FS_TGSI;
   if (!strcmp(env, "nir"))
      return I915_FS_NIR;
   return I915_FS_BOTH;
}

static void
i915_populate_fs_metadata(struct i915_fragment_shader *ifs, nir_shader *s)
{
   ifs->num_inputs = 0;
   ifs->writes_z = s->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH);

   nir_foreach_shader_in_variable(var, s) {
      unsigned sem_name, sem_index;
      tgsi_get_gl_varying_semantic((gl_varying_slot)var->data.location, true,
                                   &sem_name, &sem_index);
      unsigned idx = ifs->num_inputs++;
      ifs->input_semantic_name[idx] = sem_name;
      ifs->input_semantic_index[idx] = sem_index;
   }
}

static void
i915_compile_tgsi(struct i915_context *i915,
                  struct i915_fragment_shader *ifs,
                  struct pipe_screen *screen,
                  nir_shader *nir_clone)
{
   ifs->state.tokens = nir_to_tgsi_options(nir_clone, screen, &ntt_options);
   ifs->state.type = PIPE_SHADER_IR_TGSI;
   tgsi_scan_shader(ifs->state.tokens, &ifs->info);
   i915_translate_fragment_program(i915, ifs);
}

static bool
corm_fs_better(const struct i915_fragment_shader *a,
               const struct i915_fragment_shader *b)
{
   if (a->nr_tex_indirect != b->nr_tex_indirect)
      return a->nr_tex_indirect < b->nr_tex_indirect;
   if (a->nr_alu_insn != b->nr_alu_insn)
      return a->nr_alu_insn < b->nr_alu_insn;
   if (a->nr_temps != b->nr_temps)
      return a->nr_temps < b->nr_temps;
   return a->num_constants < b->num_constants;
}

static const char *
corm_win_reason(const struct i915_fragment_shader *winner,
                const struct i915_fragment_shader *loser,
                char *buf, size_t len)
{
   if (!loser) {
      snprintf(buf, len, "only");
      return buf;
   }
   int da = (int)winner->nr_alu_insn - (int)loser->nr_alu_insn;
   int dp = (int)winner->nr_tex_indirect - (int)loser->nr_tex_indirect;
   int dt = (int)winner->nr_temps - (int)loser->nr_temps;
   if (dp != 0)
      snprintf(buf, len, "%+d phase", dp);
   else if (da != 0)
      snprintf(buf, len, "%+d alu", da);
   else if (dt != 0)
      snprintf(buf, len, "%+d temps", dt);
   else if ((int)winner->num_constants != (int)loser->num_constants)
      snprintf(buf, len, "%+d const",
               (int)winner->num_constants - (int)loser->num_constants);
   else if (winner->program_len == loser->program_len &&
            !memcmp(winner->program, loser->program,
                    winner->program_len * sizeof(uint32_t)))
      snprintf(buf, len, "identical");
   else
      snprintf(buf, len, "tied");
   return buf;
}

static void *
i915_create_fs_state(struct pipe_context *pipe,
                     const struct pipe_shader_state *templ)
{
   struct i915_context *i915 = i915_context(pipe);
   struct i915_fragment_shader *ifs = CALLOC_STRUCT(i915_fragment_shader);
   if (!ifs)
      return NULL;

   ifs->draw_data = draw_create_fragment_shader(i915->draw, templ);

   if (templ->type == PIPE_SHADER_IR_TGSI) {
      ifs->state.tokens = tgsi_dup_tokens(templ->tokens);
      ifs->state.type = PIPE_SHADER_IR_TGSI;
      ifs->internal = i915->no_log_program_errors;
      tgsi_scan_shader(ifs->state.tokens, &ifs->info);
      i915_translate_fragment_program(i915, ifs);
      return ifs;
   }

   assert(templ->type == PIPE_SHADER_IR_NIR);
   nir_shader *s = templ->ir.nir;
   ifs->internal = s->info.internal;

   bool debug = I915_DBG_ON(DBG_FS) &&
                (!s->info.internal || NIR_DEBUG(PRINT_INTERNAL));

   char *msg = i915_check_control_flow(s);
   if (msg) {
      if (debug) {
         mesa_logi("failing shader:");
         nir_log_shaderi(s);
      }
      if (templ->report_compile_error) {
         ((struct pipe_shader_state *)templ)->error_message = strdup(msg);
         ralloc_free(s);
         i915_delete_fs_state(NULL, ifs);
         return NULL;
      }
   }

   static enum i915_fs_mode fs_mode = -1;
   if (fs_mode == (enum i915_fs_mode)-1)
      fs_mode = i915_get_fs_mode();

   bool try_nir = (fs_mode == I915_FS_NIR || fs_mode == I915_FS_BOTH);
   bool try_tgsi = (fs_mode == I915_FS_TGSI || fs_mode == I915_FS_BOTH);

   struct i915_fragment_shader tgsi_fs = {0};

   unsigned num_corm_variants = 1u << CORM_NUM_FLAGS;
   struct i915_fragment_shader nir_results[1u << CORM_NUM_FLAGS];
   int best_nir = -1;

   if (try_nir) {
      nir_shader *nir_s = nir_shader_clone(NULL, s);
      NIR_PASS(_, nir_s, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
               type_size, (nir_lower_io_options)0);
      NIR_PASS(_, nir_s, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(_, nir_s, nir_opt_vectorize, i915_vectorize_filter, NULL);
      NIR_PASS(_, nir_s, nir_lower_bool_to_float, false);
      NIR_PASS(_, nir_s, nir_shader_lower_instructions, lower_fsqrt_filter,
               lower_fsqrt_impl, NULL);
      NIR_PASS(_, nir_s, nir_opt_copy_prop);
      NIR_PASS(_, nir_s, nir_opt_cse);
      NIR_PASS(_, nir_s, nir_opt_dce);
      NIR_PASS(_, nir_s, nir_opt_algebraic);
      NIR_PASS(_, nir_s, nir_opt_algebraic_late);
      NIR_PASS(_, nir_s, nir_opt_dce);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir_s));

      for (unsigned v = 0; v < num_corm_variants; v++) {
         struct corm_compile_opts opts = { .flags = v };
         nir_shader *variant_nir = nir_shader_clone(NULL, nir_s);
         if (v & CORM_LATE_SCALAR) {
            NIR_PASS(_, variant_nir, nir_lower_alu_to_scalar, NULL, NULL);
            NIR_PASS(_, variant_nir, nir_opt_copy_prop);
            NIR_PASS(_, variant_nir, nir_opt_algebraic);
            NIR_PASS(_, variant_nir, nir_opt_dce);
            nir_index_ssa_defs(nir_shader_get_entrypoint(variant_nir));
         }
         memset(&nir_results[v], 0, sizeof(nir_results[v]));
         i915_populate_fs_metadata(&nir_results[v], variant_nir);
         i915_translate_fragment_program_nir(i915, &nir_results[v],
                                            variant_nir, &opts);
         ralloc_free(variant_nir);

         bool ok = !nir_results[v].error || !nir_results[v].error[0];
         if (ok && (best_nir < 0 ||
                    corm_fs_better(&nir_results[v], &nir_results[best_nir])))
            best_nir = v;
      }

      ralloc_free(nir_s);
   }

   if (try_tgsi) {
      i915_compile_tgsi(i915, &tgsi_fs, pipe->screen, s);
   } else {
      ralloc_free(s);
   }

   bool nir_ok = best_nir >= 0;
   bool tgsi_ok = try_tgsi && (!tgsi_fs.error || !tgsi_fs.error[0]);
   struct i915_fragment_shader *best_nir_fs = nir_ok ? &nir_results[best_nir] : NULL;

   bool use_nir;
   if (nir_ok && tgsi_ok)
      use_nir = !corm_fs_better(&tgsi_fs, best_nir_fs);
   else
      use_nir = nir_ok;

   if (debug && try_nir && try_tgsi) {
      for (unsigned v = 0; v < num_corm_variants; v++) {
         bool ok = !nir_results[v].error || !nir_results[v].error[0];
         mesa_logi("  NIR[%02x]: %s (%d ALU, %d phase, %d temps)%s",
                   v,
                   ok ? "ok" : "FAIL",
                   nir_results[v].nr_alu_insn,
                   nir_results[v].nr_tex_indirect,
                   nir_results[v].nr_temps,
                   (int)v == best_nir ? " *" : "");
      }
      mesa_logi("  TGSI: %s (%d ALU, %d phase, %d temps)",
                tgsi_ok ? "ok" : "FAIL",
                tgsi_ok ? tgsi_fs.nr_alu_insn : 0,
                tgsi_ok ? tgsi_fs.nr_tex_indirect : 0,
                tgsi_ok ? tgsi_fs.nr_temps : 0);
      mesa_logi("  -> %s%s", use_nir ? "NIR" : "TGSI",
                use_nir ? (corm_fs_better(best_nir_fs, &tgsi_fs)
                           ? " (better)" : " (tied)") : "");
   }

   /* Free non-winning NIR variants */
   if (try_nir) {
      for (unsigned v = 0; v < num_corm_variants; v++) {
         if ((int)v != best_nir) {
            FREE(nir_results[v].program);
            ralloc_free(nir_results[v].error);
         }
      }
   }

   struct i915_fragment_shader *winner, *loser = NULL;
   struct i915_fragment_shader nir_loser_copy = {0};
   if (use_nir) {
      winner = best_nir_fs;
      loser = tgsi_ok ? &tgsi_fs : NULL;
   } else {
      winner = &tgsi_fs;
      if (best_nir_fs) {
         nir_loser_copy = *best_nir_fs;
         nir_loser_copy.program = NULL;
         loser = &nir_loser_copy;
         FREE(best_nir_fs->program);
         ralloc_free(best_nir_fs->error);
      }
   }

   if (i915 && !ifs->internal) {
      bool neither = (winner->nr_alu_insn + winner->nr_tex_insn) == 0;
      char reason[32];
      if (neither)
         snprintf(reason, sizeof(reason), "neither");
      else
         corm_win_reason(winner, loser, reason, sizeof(reason));
      util_debug_message(
         &i915->debug, SHADER_INFO,
         "%s shader [%s, %s]: %d instructions, %d alu, %d tex, "
         "%d tex_indirect, %d temps, %d const",
         _mesa_shader_stage_to_abbrev(MESA_SHADER_FRAGMENT),
         neither ? "FAIL" : use_nir ? "NIR" : "TGSI", reason,
         winner->nr_alu_insn + winner->nr_tex_insn,
         winner->nr_alu_insn, winner->nr_tex_insn, winner->nr_tex_indirect,
         winner->nr_temps, winner->num_constants);
   }

   ifs->program = winner->program;
   ifs->program_len = winner->program_len;
   ifs->nr_alu_insn = winner->nr_alu_insn;
   ifs->nr_tex_insn = winner->nr_tex_insn;
   ifs->nr_tex_indirect = winner->nr_tex_indirect;
   ifs->nr_temps = winner->nr_temps;
   ifs->num_constants = winner->num_constants;
   memcpy(ifs->constants, winner->constants, sizeof(ifs->constants));
   memcpy(ifs->constant_flags, winner->constant_flags,
          sizeof(ifs->constant_flags));
   memcpy(ifs->texcoords, winner->texcoords, sizeof(ifs->texcoords));
   ifs->reads_pntc = winner->reads_pntc;
   ifs->writes_z = winner->writes_z;
   ifs->num_inputs = winner->num_inputs;
   memcpy(ifs->input_semantic_name, winner->input_semantic_name,
          sizeof(ifs->input_semantic_name));
   memcpy(ifs->input_semantic_index, winner->input_semantic_index,
          sizeof(ifs->input_semantic_index));
   if (winner->error)
      ifs->error = winner->error;

   /* The loser's info may be in use (TGSI path populates ifs->info) */
   if (try_tgsi)
      ifs->info = tgsi_fs.info;

   if (loser) {
      FREE(loser->program);
      ralloc_free(loser->error);
   }
   if (!use_nir && try_tgsi) {
      /* TGSI won — tokens are in tgsi_fs via i915_compile_tgsi.
       * We need them for ifs->state for draw's FS pipeline. */
      ifs->state = tgsi_fs.state;
   } else if (try_tgsi) {
      FREE((void *)tgsi_fs.state.tokens);
   }

   if (ifs->error && templ->report_compile_error) {
      ((struct pipe_shader_state *)templ)->error_message = strdup(ifs->error);
      i915_delete_fs_state(NULL, ifs);
      return NULL;
   }

   return ifs;
}

static void
i915_bind_fs_state(struct pipe_context *pipe, void *shader)
{
   struct i915_context *i915 = i915_context(pipe);

   if (i915->fs == shader)
      return;

   i915->fs = (struct i915_fragment_shader *)shader;

   draw_bind_fragment_shader(i915->draw,
                             (i915->fs ? i915->fs->draw_data : NULL));

   /* Tell draw if we need to do point sprites so we can get PNTC. */
   if (i915->fs)
      draw_wide_point_sprites(i915->draw, i915->fs->reads_pntc);

   i915->dirty |= I915_NEW_FS;
}

static void
i915_delete_fs_state(struct pipe_context *pipe, void *shader)
{
   struct i915_context *i915 = i915_context(pipe);
   struct i915_fragment_shader *ifs = (struct i915_fragment_shader *)shader;

   ralloc_free(ifs->error);
   FREE(ifs->program);
   ifs->program = NULL;
   FREE((struct tgsi_token *)ifs->state.tokens);
   ifs->state.tokens = NULL;

   if (ifs->draw_data) {
      if (likely(i915))
         draw_delete_fragment_shader(i915->draw, ifs->draw_data);
      else
         draw_delete_fragment_shader(NULL, ifs->draw_data);
   }

   ifs->program_len = 0;

   FREE(ifs);
}

static void *
i915_create_vs_state(struct pipe_context *pipe,
                     const struct pipe_shader_state *templ)
{
   struct i915_context *i915 = i915_context(pipe);
   void *vertex_shader;

   struct pipe_shader_state from_nir = {PIPE_SHADER_IR_TGSI};
   if (templ->type == PIPE_SHADER_IR_NIR) {
      nir_shader *s = templ->ir.nir;

      NIR_PASS(_, s, nir_lower_point_size, 1.0, 255.0);

      /* The gallivm draw path doesn't support non-native-integers NIR shaders,
       * st/mesa does native-integers for the screen as a whole rather than
       * per-stage, and i915 FS can't do native integers.  So, convert to TGSI,
       * where the draw path *does* support non-native-integers.
       */
      from_nir.tokens = nir_to_tgsi(s, pipe->screen);
      templ = &from_nir;
   }

   vertex_shader = draw_create_vertex_shader(i915->draw, templ);

   FREE((void *)from_nir.tokens);

   return vertex_shader;
}

static void
i915_bind_vs_state(struct pipe_context *pipe, void *shader)
{
   struct i915_context *i915 = i915_context(pipe);

   if (i915->vs == shader)
      return;

   i915->vs = shader;

   /* just pass-through to draw module */
   draw_bind_vertex_shader(i915->draw, (struct draw_vertex_shader *)shader);

   i915->dirty |= I915_NEW_VS;
}

static void
i915_delete_vs_state(struct pipe_context *pipe, void *shader)
{
   struct i915_context *i915 = i915_context(pipe);

   /* just pass-through to draw module */
   draw_delete_vertex_shader(i915->draw, (struct draw_vertex_shader *)shader);
}

static void
i915_set_constant_buffer(struct pipe_context *pipe,
                         mesa_shader_stage shader, uint32_t index,
                         const struct pipe_constant_buffer *cb)
{
   struct i915_context *i915 = i915_context(pipe);
   struct pipe_resource *buf = cb ? cb->buffer : NULL;
   unsigned new_num = 0;
   bool diff = true;

   /* XXX don't support geom shaders now */
   if (shader == MESA_SHADER_GEOMETRY)
      return;

   if (cb && cb->user_buffer) {
      buf = i915_user_buffer_create(pipe->screen, (void *)cb->user_buffer,
                                    cb->buffer_size, PIPE_BIND_CONSTANT_BUFFER);
   }

   /* if we have a new buffer compare it with the old one */
   if (buf) {
      struct i915_buffer *ibuf = i915_buffer(buf);
      struct pipe_resource *old_buf = i915->constants[shader];
      struct i915_buffer *old = old_buf ? i915_buffer(old_buf) : NULL;
      unsigned old_num = i915->current.num_user_constants[shader];

      new_num = ibuf->b.width0 / 4 * sizeof(float);

      if (old_num == new_num) {
         if (old_num == 0)
            diff = false;
#if 0
         /* XXX no point in running this code since st/mesa only uses user buffers */
         /* Can't compare the buffer data since they are userbuffers */
         else if (old && old->free_on_destroy)
            diff = memcmp(old->data, ibuf->data, ibuf->b.width0);
#else
         (void)old;
#endif
      }
   } else {
      diff = i915->current.num_user_constants[shader] != 0;
   }

   pipe_resource_reference(&i915->constants[shader], buf);
   i915->current.num_user_constants[shader] = new_num;

   if (diff)
      i915->dirty |= shader == MESA_SHADER_VERTEX ? I915_NEW_VS_CONSTANTS
                                                  : I915_NEW_FS_CONSTANTS;

   if (cb && cb->user_buffer) {
      pipe_resource_reference(&buf, NULL);
   }
}

static void
i915_set_sampler_views(struct pipe_context *pipe, mesa_shader_stage shader,
                       unsigned start, unsigned num,
                       unsigned unbind_num_trailing_slots,
                       struct pipe_sampler_view **views)
{
   if (shader != MESA_SHADER_FRAGMENT) {
      /* No support for VS samplers, because it would mean accessing the
       * write-combined maps of the textures, which is very slow.  VS samplers
       * are not a required feature of GL2.1 or GLES2.
       */
      assert(num == 0);
      return;
   }
   struct i915_context *i915 = i915_context(pipe);
   uint32_t i;

   assert(num <= PIPE_MAX_SAMPLERS);

   /* Check for no-op */
   if (views && num == i915->num_fragment_sampler_views &&
       !memcmp(i915->fragment_sampler_views, views,
               num * sizeof(struct pipe_sampler_view *))) {
      return;
   }

   for (i = 0; i < num; i++) {
      pipe_sampler_view_reference(&i915->fragment_sampler_views[i],
                                    views[i]);
   }

   for (i = num; i < i915->num_fragment_sampler_views; i++)
      pipe_sampler_view_reference(&i915->fragment_sampler_views[i], NULL);

   i915->num_fragment_sampler_views = num;

   i915->dirty |= I915_NEW_SAMPLER_VIEW;
}

struct pipe_sampler_view *
i915_create_sampler_view_custom(struct pipe_context *pipe,
                                struct pipe_resource *texture,
                                const struct pipe_sampler_view *templ,
                                unsigned width0, unsigned height0)
{
   struct pipe_sampler_view *view = CALLOC_STRUCT(pipe_sampler_view);

   if (view) {
      *view = *templ;
      view->reference.count = 1;
      view->texture = NULL;
      pipe_resource_reference(&view->texture, texture);
      view->context = pipe;
   }

   return view;
}

static struct pipe_sampler_view *
i915_create_sampler_view(struct pipe_context *pipe,
                         struct pipe_resource *texture,
                         const struct pipe_sampler_view *templ)
{
   struct pipe_sampler_view *view = CALLOC_STRUCT(pipe_sampler_view);

   if (view) {
      *view = *templ;
      view->reference.count = 1;
      view->texture = NULL;
      pipe_resource_reference(&view->texture, texture);
      view->context = pipe;
   }

   return view;
}

static void
i915_sampler_view_destroy(struct pipe_context *pipe,
                          struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

void
i915_framebuffer_init(struct pipe_context *pctx, const struct pipe_framebuffer_state *fb, struct pipe_surface **cbufs, struct pipe_surface **zsbuf)
{
   if (fb) {
      for (unsigned i = 0; i < fb->nr_cbufs; i++) {
         if (cbufs[i] && pipe_surface_equal(&fb->cbufs[i], cbufs[i]))
            continue;

         struct pipe_surface *psurf = fb->cbufs[i].texture ? i915_create_surface(pctx, fb->cbufs[i].texture, &fb->cbufs[i]) : NULL;
         if (cbufs[i])
            i915_surface_destroy(pctx, cbufs[i]);
         cbufs[i] = psurf;
      }

      for (unsigned i = fb->nr_cbufs; i < 1; i++) {
         if (cbufs[i])
            i915_surface_destroy(pctx, cbufs[i]);
         cbufs[i] = NULL;
      }

      if (*zsbuf && pipe_surface_equal(&fb->zsbuf, *zsbuf))
         return;
      struct pipe_surface *zsurf = fb->zsbuf.texture ? i915_create_surface(pctx, fb->zsbuf.texture, &fb->zsbuf) : NULL;
      if (*zsbuf)
         i915_surface_destroy(pctx, *zsbuf);
      *zsbuf = zsurf;
   } else {
      for (unsigned i = 0; i < 1; i++) {
         if (cbufs[i])
            i915_surface_destroy(pctx, cbufs[i]);
         cbufs[i] = NULL;
      }
      if (*zsbuf)
         i915_surface_destroy(pctx, *zsbuf);
      *zsbuf = NULL;
   }
}

static void
i915_set_framebuffer_state(struct pipe_context *pipe,
                           const struct pipe_framebuffer_state *fb)
{
   struct i915_context *i915 = i915_context(pipe);

   i915_framebuffer_init(pipe, fb, i915->fb_cbufs, &i915->fb_zsbuf);
   util_copy_framebuffer_state(&i915->framebuffer, fb);
   if (fb->nr_cbufs) {
      struct i915_surface *surf = i915_surface(i915->fb_cbufs[0]);
      if (i915->current.fixup_swizzle != surf->oc_swizzle) {
         i915->current.fixup_swizzle = surf->oc_swizzle;
         memcpy(i915->current.color_swizzle, surf->color_swizzle,
                sizeof(surf->color_swizzle));
         i915->dirty |= I915_NEW_COLOR_SWIZZLE;
      }
   }
   if (fb->zsbuf.texture)
      draw_set_zs_format(i915->draw, fb->zsbuf.format);

   i915->dirty |= I915_NEW_FRAMEBUFFER;
}

static void
i915_set_clip_state(struct pipe_context *pipe,
                    const struct pipe_clip_state *clip)
{
   struct i915_context *i915 = i915_context(pipe);

   i915->clip = *clip;

   draw_set_clip_state(i915->draw, clip);

   i915->dirty |= I915_NEW_CLIP;
}

/* Called when gallium frontends notice changes to the viewport
 * matrix:
 */
static void
i915_set_viewport_states(struct pipe_context *pipe, unsigned start_slot,
                         unsigned num_viewports,
                         const struct pipe_viewport_state *viewport)
{
   struct i915_context *i915 = i915_context(pipe);

   i915->viewport = *viewport; /* struct copy */

   /* pass the viewport info to the draw module */
   draw_set_viewport_states(i915->draw, start_slot, num_viewports,
                            &i915->viewport);

   i915->dirty |= I915_NEW_VIEWPORT;
}

static void *
i915_create_rasterizer_state(struct pipe_context *pipe,
                             const struct pipe_rasterizer_state *rasterizer)
{
   struct i915_rasterizer_state *cso = CALLOC_STRUCT(i915_rasterizer_state);

   cso->templ = *rasterizer;
   cso->light_twoside = rasterizer->light_twoside;
   cso->ds[0].u = _3DSTATE_DEPTH_OFFSET_SCALE;
   cso->ds[1].f = rasterizer->offset_scale;
   if (rasterizer->poly_stipple_enable) {
      cso->st |= ST1_ENABLE;
   }

   if (rasterizer->scissor)
      cso->sc[0] = _3DSTATE_SCISSOR_ENABLE_CMD | ENABLE_SCISSOR_RECT;
   else
      cso->sc[0] = _3DSTATE_SCISSOR_ENABLE_CMD | DISABLE_SCISSOR_RECT;

   switch (rasterizer->cull_face) {
   case PIPE_FACE_NONE:
      cso->LIS4 |= S4_CULLMODE_NONE;
      break;
   case PIPE_FACE_FRONT:
      if (rasterizer->front_ccw)
         cso->LIS4 |= S4_CULLMODE_CCW;
      else
         cso->LIS4 |= S4_CULLMODE_CW;
      break;
   case PIPE_FACE_BACK:
      if (rasterizer->front_ccw)
         cso->LIS4 |= S4_CULLMODE_CW;
      else
         cso->LIS4 |= S4_CULLMODE_CCW;
      break;
   case PIPE_FACE_FRONT_AND_BACK:
      cso->LIS4 |= S4_CULLMODE_BOTH;
      break;
   }

   {
      int line_width = CLAMP((int)(rasterizer->line_width * 2), 1, 0xf);

      cso->LIS4 |= line_width << S4_LINE_WIDTH_SHIFT;

      if (rasterizer->line_smooth)
         cso->LIS4 |= S4_LINE_ANTIALIAS_ENABLE;
   }

   {
      int point_size = CLAMP((int)rasterizer->point_size, 1, 0xff);

      cso->LIS4 |= point_size << S4_POINT_WIDTH_SHIFT;
   }

   if (rasterizer->flatshade) {
      cso->LIS4 |=
         (S4_FLATSHADE_ALPHA | S4_FLATSHADE_COLOR | S4_FLATSHADE_SPECULAR);
   }

   if (!rasterizer->flatshade_first)
      cso->LIS6 |= (2 << S6_TRISTRIP_PV_SHIFT);

   cso->LIS7 = fui(rasterizer->offset_units);

   return cso;
}

static void
i915_bind_rasterizer_state(struct pipe_context *pipe, void *raster)
{
   struct i915_context *i915 = i915_context(pipe);

   if (i915->rasterizer == raster)
      return;

   i915->rasterizer = (struct i915_rasterizer_state *)raster;

   /* pass-through to draw module */
   draw_set_rasterizer_state(
      i915->draw, (i915->rasterizer ? &(i915->rasterizer->templ) : NULL),
      raster);

   i915->dirty |= I915_NEW_RASTERIZER;
}

static void
i915_delete_rasterizer_state(struct pipe_context *pipe, void *raster)
{
   FREE(raster);
}

static void
i915_set_vertex_buffers(struct pipe_context *pipe, unsigned count,
                        const struct pipe_vertex_buffer *buffers)
{
   struct i915_context *i915 = i915_context(pipe);
   struct draw_context *draw = i915->draw;

   assert(count <= PIPE_MAX_ATTRIBS);

   util_set_vertex_buffers_count(draw->pt.vertex_buffer,
                                 &draw->pt.nr_vertex_buffers, buffers, count);
}

static void *
i915_create_vertex_elements_state(struct pipe_context *pipe, unsigned count,
                                  const struct pipe_vertex_element *attribs)
{
   struct i915_velems_state *velems;
   assert(count <= PIPE_MAX_ATTRIBS);
   velems =
      (struct i915_velems_state *)MALLOC(sizeof(struct i915_velems_state));
   if (velems) {
      velems->count = count;
      memcpy(velems->velem, attribs, sizeof(*attribs) * count);
   }
   return velems;
}

static void
i915_bind_vertex_elements_state(struct pipe_context *pipe, void *velems)
{
   struct i915_context *i915 = i915_context(pipe);
   struct i915_velems_state *i915_velems = (struct i915_velems_state *)velems;

   if (i915->velems == velems)
      return;

   i915->velems = velems;

   /* pass-through to draw module */
   if (i915_velems) {
      draw_set_vertex_elements(i915->draw, i915_velems->count,
                               i915_velems->velem);
   }
}

static void
i915_delete_vertex_elements_state(struct pipe_context *pipe, void *velems)
{
   FREE(velems);
}

static void
i915_set_sample_mask(struct pipe_context *pipe, unsigned sample_mask)
{
}

void
i915_init_state_functions(struct i915_context *i915)
{
   i915->base.create_blend_state = i915_create_blend_state;
   i915->base.bind_blend_state = i915_bind_blend_state;
   i915->base.delete_blend_state = i915_delete_blend_state;

   i915->base.create_sampler_state = i915_create_sampler_state;
   i915->base.bind_sampler_states = i915_bind_sampler_states;
   i915->base.delete_sampler_state = i915_delete_sampler_state;

   i915->base.create_depth_stencil_alpha_state =
      i915_create_depth_stencil_state;
   i915->base.bind_depth_stencil_alpha_state = i915_bind_depth_stencil_state;
   i915->base.delete_depth_stencil_alpha_state =
      i915_delete_depth_stencil_state;

   i915->base.create_rasterizer_state = i915_create_rasterizer_state;
   i915->base.bind_rasterizer_state = i915_bind_rasterizer_state;
   i915->base.delete_rasterizer_state = i915_delete_rasterizer_state;
   i915->base.create_fs_state = i915_create_fs_state;
   i915->base.bind_fs_state = i915_bind_fs_state;
   i915->base.delete_fs_state = i915_delete_fs_state;
   i915->base.create_vs_state = i915_create_vs_state;
   i915->base.bind_vs_state = i915_bind_vs_state;
   i915->base.delete_vs_state = i915_delete_vs_state;
   i915->base.create_vertex_elements_state = i915_create_vertex_elements_state;
   i915->base.bind_vertex_elements_state = i915_bind_vertex_elements_state;
   i915->base.delete_vertex_elements_state = i915_delete_vertex_elements_state;

   i915->base.set_blend_color = i915_set_blend_color;
   i915->base.set_stencil_ref = i915_set_stencil_ref;
   i915->base.set_clip_state = i915_set_clip_state;
   i915->base.set_sample_mask = i915_set_sample_mask;
   i915->base.set_constant_buffer = i915_set_constant_buffer;
   i915->base.set_framebuffer_state = i915_set_framebuffer_state;

   i915->base.set_polygon_stipple = i915_set_polygon_stipple;
   i915->base.set_scissor_states = i915_set_scissor_states;
   i915->base.set_sampler_views = i915_set_sampler_views;
   i915->base.create_sampler_view = i915_create_sampler_view;
   i915->base.sampler_view_destroy = i915_sampler_view_destroy;
   i915->base.sampler_view_release = u_default_sampler_view_release;
   i915->base.resource_release = u_default_resource_release;
   i915->base.set_viewport_states = i915_set_viewport_states;
   i915->base.set_vertex_buffers = i915_set_vertex_buffers;
}
