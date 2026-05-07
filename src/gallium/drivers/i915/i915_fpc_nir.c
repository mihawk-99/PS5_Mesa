/*
 * Copyright 2025 Red Hat, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "tgsi/tgsi_from_mesa.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "util/u_memory.h"

#include "i915_context.h"
#include "i915_debug.h"
#include "i915_debug_private.h"
#include "i915_fpc.h"
#include "i915_reg.h"

struct nir_to_i915 {
   struct corm_compile_opts opts;
   struct i915_fp_compile *p;
   struct i915_fragment_shader *ifs;

   uint32_t *ureg_map;
   uint32_t **def_csr;
   float *deferred_const;
   unsigned ureg_map_size;

   int *last_use;
   bool *is_temp;
   int ip;
};

static bool
mark_last_use_cb(nir_src *src, void *state)
{
   struct nir_to_i915 *c = state;
   if (src->ssa->index < c->ureg_map_size)
      c->last_use[src->ssa->index] = c->ip;
   return true;
}

static void
compute_last_use(struct nir_to_i915 *c, nir_function_impl *impl)
{
   c->ip = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_foreach_src(instr, mark_last_use_cb, c);
         c->ip++;
      }
   }
}

static bool
release_if_last_use_cb(nir_src *src, void *state)
{
   struct nir_to_i915 *c = state;
   unsigned idx = src->ssa->index;
   if (idx < c->ureg_map_size && c->last_use[idx] == c->ip &&
       c->is_temp[idx]) {
      uint32_t ureg = c->ureg_map[idx];
      i915_release_temp(c->p, GET_UREG_NR(ureg));
   }
   return true;
}

static void
release_dead_temps(struct nir_to_i915 *c, nir_instr *instr)
{
   nir_foreach_src(instr, release_if_last_use_cb, c);
}

static void
set_ureg(struct nir_to_i915 *c, nir_def *def, uint32_t ureg)
{
   assert(def->index < c->ureg_map_size);
   c->ureg_map[def->index] = ureg;
}

static bool
is_deferred(struct nir_to_i915 *c, unsigned ssa_index)
{
   return c->ureg_map[ssa_index] == UREG_BAD;
}

static uint32_t
resolve_const(struct nir_to_i915 *c, unsigned ssa_index, int preferred_reg)
{
   uint32_t ureg = i915_emit_const1f_prefer(c->p,
                                            c->deferred_const[ssa_index],
                                            preferred_reg);
   c->ureg_map[ssa_index] = ureg;
   return ureg;
}

static uint32_t
src_ureg(struct nir_to_i915 *c, nir_src *src)
{
   assert(src->ssa->index < c->ureg_map_size);
   if (c->ureg_map[src->ssa->index] == UREG_BAD)
      resolve_const(c, src->ssa->index, -1);
   return c->ureg_map[src->ssa->index];
}

static uint32_t
alu_src_ureg(struct nir_to_i915 *c, nir_alu_src *src)
{
   uint32_t ureg = src_ureg(c, &src->src);
   return swizzle(ureg,
                  src->swizzle[0], src->swizzle[1],
                  src->swizzle[2], src->swizzle[3]);
}

static uint32_t
def_mask(nir_def *def)
{
   uint32_t mask = 0;
   if (def->num_components >= 1) mask |= A0_DEST_CHANNEL_X;
   if (def->num_components >= 2) mask |= A0_DEST_CHANNEL_Y;
   if (def->num_components >= 3) mask |= A0_DEST_CHANNEL_Z;
   if (def->num_components >= 4) mask |= A0_DEST_CHANNEL_W;
   return mask;
}

static uint32_t
writemask_to_mask(unsigned wm)
{
   uint32_t mask = 0;
   if (wm & 1) mask |= A0_DEST_CHANNEL_X;
   if (wm & 2) mask |= A0_DEST_CHANNEL_Y;
   if (wm & 4) mask |= A0_DEST_CHANNEL_Z;
   if (wm & 8) mask |= A0_DEST_CHANNEL_W;
   return mask;
}

static uint32_t
get_texcoord_mapping(struct i915_fragment_shader *fs,
                     unsigned semantic, int index)
{
   for (int i = 0; i < I915_TEX_UNITS; i++) {
      if (fs->texcoords[i].semantic == -1) {
         fs->texcoords[i].semantic = semantic;
         fs->texcoords[i].index = index;
         return i;
      }
      if (fs->texcoords[i].semantic == (int)semantic &&
          fs->texcoords[i].index == index)
         return i;
   }
   return 0;
}

static uint32_t
emit_input(struct nir_to_i915 *c, unsigned location)
{
   struct i915_fp_compile *p = c->p;
   struct i915_fragment_shader *ifs = c->ifs;
   unsigned sem_name, sem_index;

   tgsi_get_gl_varying_semantic((gl_varying_slot)location, true,
                                &sem_name, &sem_index);

   switch (sem_name) {
   case TGSI_SEMANTIC_GENERIC:
   case TGSI_SEMANTIC_TEXCOORD:
   case TGSI_SEMANTIC_PCOORD:
   case TGSI_SEMANTIC_POSITION: {
      if (sem_name == TGSI_SEMANTIC_PCOORD)
         ifs->reads_pntc = true;
      int tc = get_texcoord_mapping(ifs, sem_name, sem_index);
      return i915_emit_decl(p, REG_TYPE_T, T_TEX0 + tc, D0_CHANNEL_ALL);
   }
   case TGSI_SEMANTIC_COLOR:
      if (sem_index == 0) {
         return i915_emit_decl(p, REG_TYPE_T, T_DIFFUSE, D0_CHANNEL_ALL);
      } else {
         return swizzle(
            i915_emit_decl(p, REG_TYPE_T, T_SPECULAR, D0_CHANNEL_XYZ),
            X, Y, Z, ONE);
      }
   case TGSI_SEMANTIC_FOG:
      return swizzle(
         i915_emit_decl(p, REG_TYPE_T, T_FOG_W, D0_CHANNEL_W),
         W, W, W, W);
   case TGSI_SEMANTIC_FACE: {
      int tc = get_texcoord_mapping(ifs, sem_name, sem_index);
      return i915_emit_decl(p, REG_TYPE_T, T_TEX0 + tc, D0_CHANNEL_X);
   }
   default:
      i915_program_error(p, "Bad input location %d (semantic %d)",
                         location, sem_name);
      return 0;
   }
}

static void
emit_load_const(struct nir_to_i915 *c, nir_load_const_instr *load)
{
   struct i915_fp_compile *p = c->p;

   switch (load->def.num_components) {
   case 1: {
      float val = load->value[0].f32;
      if ((c->opts.flags & CORM_DEFERRED_CONST) &&
          val != 0.0f && val != 1.0f && val != -1.0f) {
         c->deferred_const[load->def.index] = val;
         set_ureg(c, &load->def, UREG_BAD);
      } else {
         set_ureg(c, &load->def, i915_emit_const1f(p, val));
      }
      break;
   }
   case 2:
      set_ureg(c, &load->def,
               i915_emit_const2f(p, load->value[0].f32,
                                 load->value[1].f32));
      break;
   case 3:
   case 4: {
      unsigned n = load->def.num_components;
      float v[4] = {
         load->value[0].f32,
         n > 1 ? load->value[1].f32 : 0.0f,
         n > 2 ? load->value[2].f32 : 0.0f,
         n > 3 ? load->value[3].f32 : 0.0f,
      };

      uint32_t ch[4] = { X, Y, Z, W };
      int ng[4] = { 0, 0, 0, 0 };
      bool all_swizzle = true;
      for (unsigned i = 0; i < n; i++) {
         if (v[i] == 0.0f)
            ch[i] = ZERO;
         else if (v[i] == 1.0f)
            ch[i] = ONE;
         else if (v[i] == -1.0f) {
            ch[i] = ONE;
            ng[i] = 1;
         } else {
            all_swizzle = false;
            break;
         }
      }

      if (all_swizzle) {
         set_ureg(c, &load->def,
                  negate(swizzle(UREG(REG_TYPE_R, 0),
                                 ch[0], ch[1], ch[2], ch[3]),
                         ng[0], ng[1], ng[2], ng[3]));
      } else {
         set_ureg(c, &load->def, i915_emit_const4fv(p, v));
      }
      break;
   }
   default:
      i915_program_error(p, "load_const with %d components",
                         load->def.num_components);
      break;
   }
}

static void
coalesce_constants(struct nir_to_i915 *c, nir_alu_instr *alu)
{
   unsigned n = nir_op_infos[alu->op].num_inputs;
   unsigned deferred[3];
   unsigned nr_deferred = 0;
   int preferred = -1;

   for (unsigned i = 0; i < n; i++) {
      unsigned idx = alu->src[i].src.ssa->index;
      if (is_deferred(c, idx)) {
         deferred[nr_deferred++] = idx;
      } else {
         uint32_t ureg = c->ureg_map[idx];
         if (GET_UREG_TYPE(ureg) == REG_TYPE_CONST && preferred < 0)
            preferred = GET_UREG_NR(ureg);
      }
   }

   if (nr_deferred == 0)
      return;

   for (unsigned i = 0; i < nr_deferred; i++) {
      uint32_t ureg = resolve_const(c, deferred[i], preferred);
      if (preferred < 0 && GET_UREG_TYPE(ureg) == REG_TYPE_CONST)
         preferred = GET_UREG_NR(ureg);
   }
}

static void
emit_alu(struct nir_to_i915 *c, nir_alu_instr *alu)
{
   struct i915_fp_compile *p = c->p;
   nir_def *def = &alu->def;

   if (def->index < c->ureg_map_size && c->ureg_map[def->index] != 0)
      return;

   uint32_t mask = def_mask(def);
   uint32_t dest = UREG(REG_TYPE_R, i915_get_temp(p));
   set_ureg(c, def, dest);
   c->is_temp[def->index] = true;

   coalesce_constants(c, alu);

   uint32_t src0 = 0, src1 = 0, src2 = 0;
   if (nir_op_infos[alu->op].num_inputs >= 1)
      src0 = alu_src_ureg(c, &alu->src[0]);
   if (nir_op_infos[alu->op].num_inputs >= 2)
      src1 = alu_src_ureg(c, &alu->src[1]);
   if (nir_op_infos[alu->op].num_inputs >= 3)
      src2 = alu_src_ureg(c, &alu->src[2]);

   uint32_t *pre_csr = p->csr;

   switch (alu->op) {
   case nir_op_mov:
   case nir_op_fcanonicalize:
   case nir_op_fneg: {
      c->is_temp[def->index] = false;
      i915_release_temp(p, GET_UREG_NR(dest));
      set_ureg(c, def, alu->op == nir_op_fneg ? negate(src0, 1, 1, 1, 1)
                                         : src0);
      unsigned src_idx = alu->src[0].src.ssa->index;
      if (c->last_use[src_idx] == c->ip)
         c->last_use[src_idx] = c->last_use[def->index];
      return;
   }
   case nir_op_fabs:
      i915_emit_arith(p, A0_MAX, dest, mask, 0,
                      src0, negate(src0, 1, 1, 1, 1), 0);
      break;
   case nir_op_fsat: {
      nir_def *src_def = alu->src[0].src.ssa;
      uint32_t *prev = c->def_csr[src_def->index];
      if (prev && list_is_singular(&src_def->uses)) {
         prev[0] |= A0_DEST_SATURATE;
         i915_release_temp(p, GET_UREG_NR(dest));
         set_ureg(c, def, src_ureg(c, &alu->src[0].src));
         c->def_csr[def->index] = prev;
         unsigned src_idx = alu->src[0].src.ssa->index;
         if (c->last_use[src_idx] == c->ip)
            c->last_use[src_idx] = c->last_use[def->index];
         return;
      }
      i915_emit_arith(p, A0_MOV, dest, mask, A0_DEST_SATURATE, src0, 0, 0);
      break;
   }
   case nir_op_fadd:
      i915_emit_arith(p, A0_ADD, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_fmul:
      i915_emit_arith(p, A0_MUL, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_ffma:
      i915_emit_arith(p, A0_MAD, dest, mask, 0, src0, src1, src2);
      break;
   case nir_op_fmin:
   case nir_op_imin:
   case nir_op_umin:
      i915_emit_arith(p, A0_MIN, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_fmax:
   case nir_op_imax:
   case nir_op_umax:
      i915_emit_arith(p, A0_MAX, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_ffloor:
      i915_emit_arith(p, A0_FLR, dest, mask, 0, src0, 0, 0);
      break;
   case nir_op_ffract:
      i915_emit_arith(p, A0_FRC, dest, mask, 0, src0, 0, 0);
      break;
   case nir_op_ftrunc:
      i915_emit_arith(p, A0_TRC, dest, mask, 0, src0, 0, 0);
      break;
   case nir_op_fceil: {
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_arith(p, A0_FLR, tmp, mask, 0,
                      negate(src0, 1, 1, 1, 1), 0, 0);
      i915_emit_arith(p, A0_MOV, dest, mask, 0,
                      negate(tmp, 1, 1, 1, 1), 0, 0);
      break;
   }
   case nir_op_frcp:
      i915_emit_arith(p, A0_RCP, dest, mask, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      break;
   case nir_op_frsq:
      i915_emit_arith(p, A0_RSQ, dest, mask, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      break;
   case nir_op_fsqrt: {
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_arith(p, A0_RSQ, tmp, A0_DEST_CHANNEL_X, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      i915_emit_arith(p, A0_MUL, dest, mask, 0,
                      src0, swizzle(tmp, X, X, X, X), 0);
      break;
   }
   case nir_op_fexp2:
      i915_emit_arith(p, A0_EXP, dest, mask, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      break;
   case nir_op_flog2:
      i915_emit_arith(p, A0_LOG, dest, mask, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      break;
   case nir_op_fdot2:
   case nir_op_fdot2_replicated:
      i915_emit_arith(p, A0_DP3, dest, mask, 0,
                      swizzle(src0, X, Y, ZERO, ZERO), src1, 0);
      break;
   case nir_op_fdot3:
   case nir_op_fdot3_replicated:
      i915_emit_arith(p, A0_DP3, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_fdot4:
   case nir_op_fdot4_replicated:
      i915_emit_arith(p, A0_DP4, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_slt:
      i915_emit_arith(p, A0_SLT, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_sge:
      i915_emit_arith(p, A0_SGE, dest, mask, 0, src0, src1, 0);
      break;
   case nir_op_seq: {
      const uint32_t zero =
         swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO);
      if ((c->opts.flags & CORM_SEQ_SNE_OPT) &&
          ((src0 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK) ||
           (src1 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK))) {
         if ((src0 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK))
            src0 = src1;
         /* x == 0  <->  -abs(x) >= 0: 2 insns instead of 3 */
         uint32_t tmp = i915_get_utemp(p);
         i915_emit_arith(p, A0_MAX, tmp, A0_DEST_CHANNEL_ALL, 0,
                         src0, negate(src0, 1, 1, 1, 1), 0);
         i915_emit_arith(p, A0_SGE, dest, mask, 0,
                         negate(tmp, 1, 1, 1, 1), zero, 0);
      } else {
         /* seq(a,b) = sge(a,b) * sge(b,a) */
         uint32_t tmp = i915_get_utemp(p);
         i915_emit_arith(p, A0_SGE, tmp, A0_DEST_CHANNEL_ALL, 0,
                         src0, src1, 0);
         i915_emit_arith(p, A0_SGE, dest, mask, 0, src1, src0, 0);
         i915_emit_arith(p, A0_MUL, dest, mask, 0, dest, tmp, 0);
      }
      break;
   }
   case nir_op_sne: {
      const uint32_t zero =
         swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO);
      if ((c->opts.flags & CORM_SEQ_SNE_OPT) &&
          ((src0 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK) ||
           (src1 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK))) {
         if ((src0 & UREG_XYZW_CHANNEL_MASK) == (zero & UREG_XYZW_CHANNEL_MASK))
            src0 = src1;
         /* x != 0  <->  -abs(x) < 0: 2 insns instead of 3 */
         uint32_t tmp = i915_get_utemp(p);
         i915_emit_arith(p, A0_MAX, tmp, A0_DEST_CHANNEL_ALL, 0,
                         src0, negate(src0, 1, 1, 1, 1), 0);
         i915_emit_arith(p, A0_SLT, dest, mask, 0,
                         negate(tmp, 1, 1, 1, 1), zero, 0);
      } else {
         /* sne(a,b) = slt(a,b) + slt(b,a) */
         uint32_t tmp = i915_get_utemp(p);
         i915_emit_arith(p, A0_SLT, tmp, A0_DEST_CHANNEL_ALL, 0,
                         src0, src1, 0);
         i915_emit_arith(p, A0_SLT, dest, mask, 0, src1, src0, 0);
         i915_emit_arith(p, A0_ADD, dest, mask, 0, dest, tmp, 0);
      }
      break;
   }
   case nir_op_fpow: {
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_arith(p, A0_LOG, tmp, A0_DEST_CHANNEL_X, 0,
                      swizzle(src0, X, X, X, X), 0, 0);
      i915_emit_arith(p, A0_MUL, tmp, A0_DEST_CHANNEL_X, 0, tmp, src1, 0);
      i915_emit_arith(p, A0_EXP, dest, mask, 0,
                      swizzle(tmp, X, X, X, X), 0, 0);
      break;
   }
   case nir_op_bcsel:
      i915_emit_arith(p, A0_CMP, dest, mask, 0,
                      negate(src0, 1, 1, 1, 1), src2, src1);
      break;
   case nir_op_fcsel_ge:
      i915_emit_arith(p, A0_CMP, dest, mask, 0, src0, src1, src2);
      break;
   case nir_op_fcsel_gt:
      i915_emit_arith(p, A0_CMP, dest, mask, 0,
                      negate(src0, 1, 1, 1, 1), src2, src1);
      break;
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      unsigned n = nir_op_infos[alu->op].num_inputs;
      uint32_t srcs[4] = { 0 };
      for (unsigned i = 0; i < n; i++)
         srcs[i] = alu_src_ureg(c, &alu->src[i]);

      bool same_reg = true;
      for (unsigned i = 1; i < n; i++) {
         if ((srcs[i] & UREG_TYPE_NR_MASK) != (srcs[0] & UREG_TYPE_NR_MASK)) {
            same_reg = false;
            break;
         }
      }

      if (same_reg) {
         uint32_t base = UREG(GET_UREG_TYPE(srcs[0]), GET_UREG_NR(srcs[0]));
         uint32_t ch[4] = { X, Y, Z, W };
         int ng[4] = { 0, 0, 0, 0 };
         for (unsigned i = 0; i < n; i++) {
            ch[i] = (srcs[i] >> UREG_CHANNEL_X_SHIFT) & 0x7;
            ng[i] = (srcs[i] >> UREG_CHANNEL_X_NEGATE_SHIFT) & 0x1;
         }
         c->is_temp[def->index] = false;
         i915_release_temp(p, GET_UREG_NR(dest));
         set_ureg(c, def, negate(swizzle(base, ch[0], ch[1], ch[2], ch[3]),
                                 ng[0], ng[1], ng[2], ng[3]));
         return;
      }

      /* If this vec's only consumer is a store_output, write directly
       * to the output register instead of going through a temp.
       * If it's a tex instruction, use a utemp to avoid phase boundaries.
       */
      if (list_is_singular(&def->uses)) {
         nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
         nir_instr *use_instr = nir_src_use_instr(use);
         if (use_instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *store =
               nir_instr_as_intrinsic(use_instr);
            if (store->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_component(store) == 0) {
               nir_io_semantics sem = nir_intrinsic_io_semantics(store);
               uint32_t out = sem.location == FRAG_RESULT_DEPTH
                  ? UREG(REG_TYPE_OD, 0) : UREG(REG_TYPE_OC, 0);
               i915_release_temp(p, GET_UREG_NR(dest));
               dest = out;
               set_ureg(c, def, dest);
            }
         } else if ((c->opts.flags & CORM_UTEMP_TEXCOORD) &&
                    use_instr->type == nir_instr_type_tex) {
            i915_release_temp(p, GET_UREG_NR(dest));
            uint32_t utemp = i915_get_utemp(p);
            dest = utemp;
            set_ureg(c, def, dest);
         }
      }

      static const uint32_t chan_mask[] = {
         A0_DEST_CHANNEL_X, A0_DEST_CHANNEL_Y,
         A0_DEST_CHANNEL_Z, A0_DEST_CHANNEL_W,
      };
      bool emitted[4] = { false };
      uint32_t ch_sel[4];
      int neg_sel[4] = { 0, 0, 0, 0 };
      for (unsigned i = 0; i < n; i++) {
         ch_sel[i] = (srcs[i] >> UREG_CHANNEL_X_SHIFT) & 0x7;
         neg_sel[i] = (srcs[i] >> UREG_CHANNEL_X_NEGATE_SHIFT) & 0x1;
      }

      /* ALU dest folding: if a vec source is a single-use ALU result in a
       * temp with identity swizzle, patch that instruction to write
       * directly into our dest with the right channel mask.
       */
      for (unsigned i = 0; i < n; i++) {
         nir_def *src_def = alu->src[i].src.ssa;
         uint32_t *prev_csr = c->def_csr[src_def->index];
         if (!prev_csr)
            continue;
         if (GET_UREG_TYPE(srcs[i]) != REG_TYPE_R)
            continue;
         unsigned nc = src_def->num_components;
         if (i + nc > n)
            continue;
         bool identity = true;
         for (unsigned j = 0; j < nc && identity; j++)
            identity = (j == 0 || alu->src[i + j].src.ssa == src_def) &&
                       (alu->src[i + j].swizzle[0] == j);
         if (!identity)
            continue;
         bool all_from_this_vec = true;
         nir_foreach_use(use, src_def) {
            if (nir_src_use_instr(use) != &alu->instr) {
               all_from_this_vec = false;
               break;
            }
         }
         if (!all_from_this_vec)
            continue;

         uint32_t fold_mask = 0;
         for (unsigned j = 0; j < nc; j++)
            fold_mask |= chan_mask[i + j];

         prev_csr[0] = (prev_csr[0] & ~(A0_DEST_CHANNEL_ALL |
                               (0x1ff << A0_DEST_NR_SHIFT))) |
                   A0_DEST(dest) | fold_mask;

         i915_release_temp(p, GET_UREG_NR(srcs[i]));
         c->ureg_map[src_def->index] = dest;
         for (unsigned j = 0; j < nc; j++)
            emitted[i + j] = true;
      }

      /* ALU consumer fusion: if this vec feeds a single binary ALU op
       * and the other ALU source is a single register, emit the ALU op
       * per-group with partial writemasks instead of MOV+ALU.
       */
      if (list_is_singular(&def->uses)) {
         nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
         nir_instr *use_instr = nir_src_use_instr(use);
         if (use_instr->type == nir_instr_type_alu) {
            nir_alu_instr *consumer = nir_instr_as_alu(use_instr);
            unsigned nargs = nir_op_infos[consumer->op].num_inputs;
            int vec_arg = -1;
            for (unsigned a = 0; a < nargs; a++) {
               if (consumer->src[a].src.ssa == def) {
                  vec_arg = a;
                  break;
               }
            }
            uint32_t hw_op = 0;
            bool can_fuse = (vec_arg >= 0);
            if (can_fuse) {
               switch (consumer->op) {
               case nir_op_fmul: hw_op = A0_MUL; break;
               case nir_op_fadd: hw_op = A0_ADD; break;
               case nir_op_ffma: hw_op = A0_MAD; break;
               case nir_op_fmin: case nir_op_imin: case nir_op_umin:
                  hw_op = A0_MIN; break;
               case nir_op_fmax: case nir_op_imax: case nir_op_umax:
                  hw_op = A0_MAX; break;
               default: can_fuse = false; break;
               }
            }
            /* check the non-vec sources are single registers */
            uint32_t other_srcs[3] = { 0, 0, 0 };
            if (can_fuse) {
               for (unsigned a = 0; a < nargs; a++) {
                  if ((int)a == vec_arg)
                     continue;
                  nir_def *od = consumer->src[a].src.ssa;
                  if (od->index >= c->ureg_map_size ||
                      c->ureg_map[od->index] == UREG_BAD) {
                     can_fuse = false;
                     break;
                  }
                  other_srcs[a] = alu_src_ureg(c, &consumer->src[a]);
               }
            }
            if (can_fuse) {
                  nir_def *cdef = &consumer->def;
                  uint32_t cdest = dest;
                  uint32_t cmask = def_mask(cdef);

                  for (unsigned i = 0; i < n; i++) {
                     if (emitted[i])
                        continue;
                     uint32_t base = UREG(GET_UREG_TYPE(srcs[i]),
                                          GET_UREG_NR(srcs[i]));
                     uint32_t group_mask = chan_mask[i];
                     uint32_t ch[4] = { X, Y, Z, W };
                     int ng[4] = { 0, 0, 0, 0 };
                     ch[i] = ch_sel[i];
                     ng[i] = neg_sel[i];
                     for (unsigned j = i + 1; j < n; j++) {
                        if (!emitted[j] &&
                            (ch_sel[j] >= SRC_ZERO ||
                             (srcs[j] & UREG_TYPE_NR_MASK) ==
                             (srcs[i] & UREG_TYPE_NR_MASK))) {
                           group_mask |= chan_mask[j];
                           ch[j] = ch_sel[j];
                           ng[j] = neg_sel[j];
                           emitted[j] = true;
                        }
                     }
                     uint32_t fused_src = negate(
                        swizzle(base, ch[0], ch[1], ch[2], ch[3]),
                        ng[0], ng[1], ng[2], ng[3]);
                     uint32_t s[3];
                     for (unsigned a = 0; a < nargs; a++)
                        s[a] = ((int)a == vec_arg) ? fused_src
                                                   : other_srcs[a];
                     i915_emit_arith(p, hw_op, cdest,
                                     group_mask & cmask, 0,
                                     s[0], nargs > 1 ? s[1] : 0,
                                     nargs > 2 ? s[2] : 0);
                     emitted[i] = true;
                  }

                  set_ureg(c, cdef, cdest);
                  c->def_csr[cdef->index] = p->csr - 3;
                  break;
            }
         }
      }

      /* Process real-register sources first, folding in any ZERO/ONE
       * const-swizzle sources that can piggyback on the same MOV.
       * Use the unswizzled base register since swizzle() composes.
       */
      for (unsigned i = 0; i < n; i++) {
         if (emitted[i] || ch_sel[i] >= SRC_ZERO)
            continue;
         uint32_t base = UREG(GET_UREG_TYPE(srcs[i]), GET_UREG_NR(srcs[i]));
         uint32_t group_mask = chan_mask[i];
         uint32_t ch[4] = { X, Y, Z, W };
         int ng[4] = { 0, 0, 0, 0 };
         ch[i] = ch_sel[i];
         ng[i] = neg_sel[i];
         for (unsigned j = i + 1; j < n; j++) {
            if (!emitted[j] &&
                (ch_sel[j] >= SRC_ZERO ||
                 (srcs[j] & UREG_TYPE_NR_MASK) ==
                 (srcs[i] & UREG_TYPE_NR_MASK))) {
               group_mask |= chan_mask[j];
               ch[j] = ch_sel[j];
               ng[j] = neg_sel[j];
               emitted[j] = true;
            }
         }
         i915_emit_arith(p, A0_MOV, dest, group_mask & mask, 0,
                         negate(swizzle(base, ch[0], ch[1], ch[2], ch[3]),
                                ng[0], ng[1], ng[2], ng[3]),
                         0, 0);
         emitted[i] = true;
      }
      /* Any remaining const-swizzle-only sources */
      for (unsigned i = 0; i < n; i++) {
         if (emitted[i])
            continue;
         uint32_t group_mask = chan_mask[i];
         uint32_t ch[4] = { X, Y, Z, W };
         int ng[4] = { 0, 0, 0, 0 };
         ch[i] = ch_sel[i];
         ng[i] = neg_sel[i];
         for (unsigned j = i + 1; j < n; j++) {
            if (!emitted[j]) {
               group_mask |= chan_mask[j];
               ch[j] = ch_sel[j];
               ng[j] = neg_sel[j];
               emitted[j] = true;
            }
         }
         i915_emit_arith(p, A0_MOV, dest, group_mask & mask, 0,
                         negate(swizzle(srcs[i], ch[0], ch[1], ch[2], ch[3]),
                                ng[0], ng[1], ng[2], ng[3]),
                         0, 0);
         emitted[i] = true;
      }
      break;
   }
   case nir_op_fsign: {
      uint32_t tmp = i915_get_utemp(p);
      const uint32_t zero = swizzle(UREG(REG_TYPE_R, 0),
                                    ZERO, ZERO, ZERO, ZERO);
      i915_emit_arith(p, A0_SLT, tmp, A0_DEST_CHANNEL_ALL, 0,
                      src0, zero, 0);
      i915_emit_arith(p, A0_SLT, dest, mask, 0, zero, src0, 0);
      i915_emit_arith(p, A0_ADD, dest, mask, 0,
                      dest, negate(tmp, 1, 1, 1, 1), 0);
      break;
   }
   default:
      i915_program_error(p, "unsupported NIR ALU op: %s",
                         nir_op_infos[alu->op].name);
      break;
   }

   if (p->csr == pre_csr + 3)
      c->def_csr[def->index] = pre_csr;

   uint32_t save = 0;
   if (GET_UREG_TYPE(dest) == REG_TYPE_U)
      save = p->utemp_flag & (1 << GET_UREG_NR(dest));
   i915_release_utemps(p);
   p->utemp_flag |= save;
}

static uint32_t
translate_tex_type(struct i915_fp_compile *p, enum glsl_sampler_dim dim)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return D0_SAMPLE_TYPE_2D;
   case GLSL_SAMPLER_DIM_3D:
      return D0_SAMPLE_TYPE_VOLUME;
   case GLSL_SAMPLER_DIM_CUBE:
      return D0_SAMPLE_TYPE_CUBE;
   default:
      i915_program_error(p, "unsupported sampler dim %d", dim);
      return D0_SAMPLE_TYPE_2D;
   }
}

static uint32_t
tex_coord_mask(nir_tex_instr *tex)
{
   uint32_t mask = TGSI_WRITEMASK_X;

   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      mask = TGSI_WRITEMASK_XY;
      break;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      mask = TGSI_WRITEMASK_XYZ;
      break;
   default:
      break;
   }

   if (tex->is_shadow)
      mask |= TGSI_WRITEMASK_Z;

   if (tex->op == nir_texop_txb)
      mask |= TGSI_WRITEMASK_W;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_projector) {
         mask |= TGSI_WRITEMASK_W;
         break;
      }
   }

   return mask;
}

static void
emit_tex(struct nir_to_i915 *c, nir_tex_instr *tex)
{
   struct i915_fp_compile *p = c->p;
   nir_def *def = &tex->def;
   uint32_t dest = UREG(REG_TYPE_R, i915_get_temp(p));
   set_ureg(c, def, dest);
   c->is_temp[def->index] = true;

   uint32_t hw_tex = translate_tex_type(p, tex->sampler_dim);
   uint32_t sampler = i915_emit_decl(p, REG_TYPE_S, tex->sampler_index, hw_tex);

   uint32_t coord = 0;
   uint32_t bias_or_proj = 0;
   uint32_t shadow = 0;
   bool has_bias = false, has_proj = false, has_shadow = false;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
         coord = src_ureg(c, &tex->src[i].src);
         break;
      case nir_tex_src_bias:
         bias_or_proj = src_ureg(c, &tex->src[i].src);
         has_bias = true;
         break;
      case nir_tex_src_projector:
         bias_or_proj = src_ureg(c, &tex->src[i].src);
         has_proj = true;
         break;
      case nir_tex_src_comparator:
         shadow = src_ureg(c, &tex->src[i].src);
         has_shadow = true;
         break;
      default:
         break;
      }
   }

   /* 1D textures: set Y = X so LOD works correctly when sampled as 2D */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_1D)
      coord = swizzle(coord, X, X, Z, W);

   /* pack bias/projector/shadow into a single coord register if needed */
   if (has_bias || has_proj || has_shadow) {
      uint32_t tmp = UREG(REG_TYPE_R, i915_get_temp(p));

      i915_emit_arith(p, A0_MOV, tmp, A0_DEST_CHANNEL_ALL, 0, coord, 0, 0);

      if (has_shadow)
         i915_emit_arith(p, A0_MOV, tmp, A0_DEST_CHANNEL_Z, 0,
                         swizzle(shadow, X, X, X, X), 0, 0);

      if (has_bias || has_proj)
         i915_emit_arith(p, A0_MOV, tmp, A0_DEST_CHANNEL_W, 0,
                         swizzle(bias_or_proj, X, X, X, X), 0, 0);

      coord = tmp;
   }

   uint32_t opcode;
   if (tex->op == nir_texop_txb) {
      opcode = T0_TEXLDB;
   } else if (has_proj) {
      opcode = T0_TEXLDP;
   } else if (tex->op == nir_texop_tex) {
      opcode = T0_TEXLD;
   } else {
      i915_program_error(p, "unsupported tex op %d", tex->op);
      return;
   }

   i915_emit_texld(p, dest, A0_DEST_CHANNEL_ALL, sampler, coord, opcode,
                   tex_coord_mask(tex));

   i915_release_utemps(p);
}

static void
emit_intrinsic(struct nir_to_i915 *c, nir_intrinsic_instr *intr)
{
   struct i915_fp_compile *p = c->p;
   struct i915_fragment_shader *ifs = c->ifs;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_input: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      unsigned comp = nir_intrinsic_component(intr);
      uint32_t reg = emit_input(c, sem.location);

      if (comp > 0) {
         reg = swizzle(reg, comp, MIN2(comp + 1, 3),
                       MIN2(comp + 2, 3), MIN2(comp + 3, 3));
      }

      set_ureg(c, &intr->def, reg);
      break;
   }

   case nir_intrinsic_store_output: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      unsigned comp = nir_intrinsic_component(intr);
      uint32_t val = src_ureg(c, &intr->src[0]);
      uint32_t wm = nir_intrinsic_write_mask(intr);
      uint32_t dest;

      if (sem.location == FRAG_RESULT_DEPTH) {
         dest = UREG(REG_TYPE_OD, 0);
      } else {
         dest = UREG(REG_TYPE_OC, 0);
      }

      /* Vec direct-output already wrote to oC/oD */
      uint32_t val_type = GET_UREG_TYPE(val);
      if (val_type == REG_TYPE_OC || val_type == REG_TYPE_OD)
         break;

      nir_def *src_def = intr->src[0].ssa;
      uint32_t *prev = c->def_csr[src_def->index];

      /* Look through identity vec (same_reg case emits no instructions).
       * Check that all uses of the underlying def come from this vec.
       */
      bool looked_through_vec = false;
      if (!prev) {
         nir_instr *def_instr = nir_def_instr_nonconst(src_def);
         if (def_instr->type == nir_instr_type_alu) {
            nir_alu_instr *vec = nir_instr_as_alu(def_instr);
            if ((vec->op == nir_op_vec4 || vec->op == nir_op_vec3 ||
                 vec->op == nir_op_vec2) &&
                list_is_singular(&src_def->uses)) {
               nir_def *inner = vec->src[0].src.ssa;
               bool all_from_vec = true;
               nir_foreach_use(use, inner) {
                  if (nir_src_use_instr(use) != def_instr) {
                     all_from_vec = false;
                     break;
                  }
               }
               if (all_from_vec) {
                  src_def = inner;
                  prev = c->def_csr[src_def->index];
                  looked_through_vec = true;
               }
            }
         }
      }

      if (prev && comp == 0 &&
          (looked_through_vec || list_is_singular(&src_def->uses))) {
         prev[0] = (prev[0] & ~(A0_DEST_CHANNEL_ALL |
                                (0x1ff << A0_DEST_NR_SHIFT))) |
                   A0_DEST(dest) | writemask_to_mask(wm);
         break;
      }

      if (comp > 0) {
         uint32_t s[4] = { X, Y, Z, W };
         for (int i = 3; i >= (int)comp; i--)
            s[i] = s[i - comp];
         for (unsigned i = 0; i < comp; i++)
            s[i] = ZERO;
         val = swizzle(val, s[0], s[1], s[2], s[3]);
         wm <<= comp;
      }

      i915_emit_arith(p, A0_MOV, dest, writemask_to_mask(wm), 0,
                      val, 0, 0);
      break;
   }

   case nir_intrinsic_load_ubo: {
      nir_src *offset_src = &intr->src[1];
      if (!nir_src_is_const(*offset_src)) {
         i915_program_error(p, "non-constant UBO offset");
         set_ureg(c, &intr->def,
                  swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
         break;
      }
      unsigned byte_offset = (unsigned)nir_src_as_float(*offset_src);
      unsigned slot = byte_offset / 16;
      unsigned comp = (byte_offset % 16) / 4;

      if (slot >= I915_MAX_CONSTANT) {
         i915_program_error(p, "UBO offset %d exceeds max constants", slot);
         set_ureg(c, &intr->def,
                  swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
         break;
      }

      for (unsigned i = 0; i < intr->def.num_components; i++)
         ifs->constant_flags[slot] |= I915_CONSTFLAG_USER_CH(comp + i);
      ifs->num_constants = MAX2(ifs->num_constants, slot + 1);

      uint32_t reg = UREG(REG_TYPE_CONST, slot);
      if (comp > 0) {
         uint32_t s[4];
         for (unsigned i = 0; i < 4; i++)
            s[i] = MIN2(comp + i, 3);
         reg = swizzle(reg, s[0], s[1], s[2], s[3]);
      }

      set_ureg(c, &intr->def, reg);
      break;
   }

   case nir_intrinsic_load_ubo_vec4: {
      nir_src *offset_src = &intr->src[1];
      if (!nir_src_is_const(*offset_src)) {
         i915_program_error(p, "non-constant UBO offset");
         set_ureg(c, &intr->def,
                  swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
         break;
      }
      unsigned slot = nir_intrinsic_base(intr) +
                      (unsigned)nir_src_as_float(*offset_src);
      unsigned comp = nir_intrinsic_component(intr);

      if (slot >= I915_MAX_CONSTANT) {
         i915_program_error(p, "UBO slot %d exceeds max constants", slot);
         set_ureg(c, &intr->def,
                  swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
         break;
      }

      for (unsigned i = 0; i < intr->def.num_components; i++)
         ifs->constant_flags[slot] |= I915_CONSTFLAG_USER_CH(comp + i);
      ifs->num_constants = MAX2(ifs->num_constants, slot + 1);

      uint32_t reg = UREG(REG_TYPE_CONST, slot);
      if (comp > 0) {
         uint32_t s[4];
         for (unsigned i = 0; i < 4; i++)
            s[i] = MIN2(comp + i, 3);
         reg = swizzle(reg, s[0], s[1], s[2], s[3]);
      }

      set_ureg(c, &intr->def, reg);
      break;
   }

   case nir_intrinsic_terminate:
   case nir_intrinsic_demote: {
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_texld(p, tmp, A0_DEST_CHANNEL_ALL, 0,
                      negate(swizzle(UREG(REG_TYPE_R, 0), ONE, ONE, ONE, ONE),
                             1, 1, 1, 1),
                      T0_TEXKILL, TGSI_WRITEMASK_X);
      i915_release_utemps(p);
      break;
   }

   case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote_if: {
      uint32_t cond = src_ureg(c, &intr->src[0]);
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_texld(p, tmp, A0_DEST_CHANNEL_ALL, 0,
                      negate(swizzle(cond, X, X, X, X), 1, 1, 1, 1),
                      T0_TEXKILL, TGSI_WRITEMASK_XYZW);
      i915_release_utemps(p);
      break;
   }

   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
      set_ureg(c, &intr->def,
               swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
      break;

   default:
      i915_program_error(p, "unsupported intrinsic: %s",
                         nir_intrinsic_infos[intr->intrinsic].name);
      break;
   }
}

static void
emit_instr(struct nir_to_i915 *c, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      emit_load_const(c, nir_instr_as_load_const(instr));
      break;
   case nir_instr_type_alu:
      emit_alu(c, nir_instr_as_alu(instr));
      break;
   case nir_instr_type_tex:
      emit_tex(c, nir_instr_as_tex(instr));
      break;
   case nir_instr_type_intrinsic:
      emit_intrinsic(c, nir_instr_as_intrinsic(instr));
      break;
   case nir_instr_type_undef: {
      nir_undef_instr *undef = nir_instr_as_undef(instr);
      set_ureg(c, &undef->def,
               swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO));
      break;
   }
   case nir_instr_type_jump:
   case nir_instr_type_deref:
      break;
   default:
      i915_program_error(c->p, "unsupported NIR instruction type %d",
                         instr->type);
      break;
   }
}

static void
fixup_depth_write(struct nir_to_i915 *c, nir_shader *s)
{
   if (!(s->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
      return;

   /* NIR writes depth to OD.X (component 0); hardware reads from OD.W */
   i915_emit_arith(c->p, A0_MOV,
                   UREG(REG_TYPE_OD, 0), A0_DEST_CHANNEL_W, 0,
                   swizzle(UREG(REG_TYPE_OD, 0), X, Y, Z, X),
                   0, 0);
}

void
i915_translate_fragment_program_nir(struct i915_context *i915,
                                    struct i915_fragment_shader *ifs,
                                    nir_shader *s,
                                    const struct corm_compile_opts *opts)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   bool debug = I915_DBG_ON(DBG_FS) &&
                (!ifs->internal || NIR_DEBUG(PRINT_INTERNAL));

   if (debug) {
      mesa_logi("NIR fragment shader:");
      nir_log_shaderi(s);
   }

   struct i915_fp_compile *p = CALLOC_STRUCT(i915_fp_compile);
   p->shader = ifs;
   p->error = ralloc_strdup(NULL, "");
   p->log_program_errors = !ifs->internal;

   ifs->num_constants = 0;
   memset(ifs->constant_flags, 0, sizeof(ifs->constant_flags));
   memset(p->register_phases, 0, sizeof(p->register_phases));

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         unsigned slot;
         if (intr->intrinsic == nir_intrinsic_load_ubo &&
             nir_src_is_const(intr->src[1])) {
            unsigned byte_offset = (unsigned)nir_src_as_float(intr->src[1]);
            slot = byte_offset / 16;
         } else if (intr->intrinsic == nir_intrinsic_load_ubo_vec4 &&
                    nir_src_is_const(intr->src[1])) {
            slot = nir_intrinsic_base(intr) +
                   (unsigned)nir_src_as_float(intr->src[1]);
         } else {
            continue;
         }
         if (slot < I915_MAX_CONSTANT) {
            ifs->constant_flags[slot] |= I915_CONSTFLAG_USER;
            ifs->num_constants = MAX2(ifs->num_constants, slot + 1);
         }
      }
   }

   for (int i = 0; i < I915_TEX_UNITS; i++)
      ifs->texcoords[i].semantic = -1;

   p->nr_tex_indirect = 1;
   p->nr_tex_insn = 0;
   p->nr_alu_insn = 0;
   p->nr_decl_insn = 0;
   p->csr = p->program;
   p->decl = p->declarations;
   p->decl_s = 0;
   p->decl_t = 0;
   p->temp_flag = ~0x0U << I915_MAX_TEMPORARY;
   p->utemp_flag = ~0x7;

   *(p->decl++) = _3DSTATE_PIXEL_SHADER_PROGRAM;

   struct nir_to_i915 c = {
      .p = p,
      .ifs = ifs,
      .opts = *opts,
      .ureg_map_size = impl->ssa_alloc,
      .ureg_map = CALLOC(impl->ssa_alloc, sizeof(uint32_t)),
      .def_csr = CALLOC(impl->ssa_alloc, sizeof(uint32_t *)),
      .deferred_const = CALLOC(impl->ssa_alloc, sizeof(float)),
      .last_use = CALLOC(impl->ssa_alloc, sizeof(int)),
      .is_temp = CALLOC(impl->ssa_alloc, sizeof(bool)),
   };

   memset(c.last_use, -1, impl->ssa_alloc * sizeof(int));
   compute_last_use(&c, impl);

   c.ip = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         emit_instr(&c, instr);
         if (p->error[0])
            break;
         release_dead_temps(&c, instr);
         c.ip++;
      }
      if (p->error[0])
         break;
   }

   if (!p->error[0])
      fixup_depth_write(&c, s);

   /* finalize */
   if (p->nr_tex_indirect > I915_MAX_TEX_INDIRECT)
      i915_program_error(p, "exceeded max tex indirect (%d/%d)",
                         p->nr_tex_indirect, I915_MAX_TEX_INDIRECT);
   if (p->nr_tex_insn > I915_MAX_TEX_INSN)
      i915_program_error(p, "exceeded max tex insn (%d/%d)",
                         p->nr_tex_insn, I915_MAX_TEX_INSN);
   if (p->nr_alu_insn > I915_MAX_ALU_INSN)
      i915_program_error(p, "exceeded max ALU insn (%d/%d)",
                         p->nr_alu_insn, I915_MAX_ALU_INSN);
   if (p->nr_decl_insn > I915_MAX_DECL_INSN)
      i915_program_error(p, "exceeded max decl insn (%d/%d)",
                         p->nr_decl_insn, I915_MAX_DECL_INSN);

   if (p->nr_alu_insn == 0 && p->nr_tex_insn == 0) {
      i915_use_passthrough_shader(ifs);
      ifs->nr_alu_insn = 1;
      goto cleanup;
   }

   ifs->nr_alu_insn = p->nr_alu_insn;
   ifs->nr_tex_insn = p->nr_tex_insn;
   ifs->nr_tex_indirect = p->nr_tex_indirect;
   ifs->nr_temps = util_bitcount(p->temp_flag);

   {
      unsigned long program_size = (unsigned long)(p->csr - p->program);
      unsigned long decl_size = (unsigned long)(p->decl - p->declarations);

      p->declarations[0] |= program_size + decl_size - 2;

      assert(!ifs->program);
      ifs->program_len = decl_size + program_size;
      ifs->program = (uint32_t *)MALLOC(ifs->program_len * sizeof(uint32_t));
      memcpy(ifs->program, p->declarations, decl_size * sizeof(uint32_t));
      memcpy(&ifs->program[decl_size], p->program,
             program_size * sizeof(uint32_t));

      if (p->error[0]) {
         /* dump the program for debugging, then replace with passthrough */
         if (debug && ifs->program_len > 2) {
            mesa_logi("FAILED program (%d ALU):", p->nr_alu_insn);
            i915_disassemble_program(ifs->program, ifs->program_len);
         }
         FREE(ifs->program);
         ifs->program = NULL;
         ifs->program_len = 0;
         i915_use_passthrough_shader(ifs);
      }
   }

cleanup:
   if (p->error[0])
      ifs->error = p->error;
   else
      ralloc_free(p->error);

   FREE(c.last_use);
   FREE(c.is_temp);
   FREE(c.deferred_const);
   FREE(c.def_csr);
   FREE(c.ureg_map);
   FREE(p);

   if (debug) {
      if (ifs->error)
         mesa_loge("%s", ifs->error);

      mesa_logi("i915 fragment shader with %d constants%s",
                ifs->num_constants, ifs->num_constants ? ":" : "");

      for (int i = 0; i < I915_MAX_CONSTANT; i++) {
         if (ifs->constant_flags[i] & 0x0f) {
            mesa_logi("\t\tC[%d] = { %f, %f, %f, %f }", i,
                      ifs->constants[i][0], ifs->constants[i][1],
                      ifs->constants[i][2], ifs->constants[i][3]);
         }
      }
      i915_disassemble_program(ifs->program, ifs->program_len);
   }
}
