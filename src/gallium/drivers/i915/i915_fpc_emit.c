/**************************************************************************
 *
 * Copyright 2003 VMware, Inc.
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

#include <stdarg.h>

#include "util/ralloc.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "i915_context.h"
#include "i915_fpc.h"
#include "i915_reg.h"

void
i915_program_error(struct i915_fp_compile *p, const char *msg, ...)
{
   va_list args;
   va_start(args, msg);
   ralloc_vasprintf_append(&p->error, msg, args);
   va_end(args);
}

static const unsigned passthrough_program[] = {
   _3DSTATE_PIXEL_SHADER_PROGRAM | ((1 * 3) - 1),
   (A0_MOV | (REG_TYPE_OC << A0_DEST_TYPE_SHIFT) | A0_DEST_CHANNEL_ALL |
    (REG_TYPE_R << A0_SRC0_TYPE_SHIFT) | (0 << A0_SRC0_NR_SHIFT)),
   ((SRC_ONE << A1_SRC0_CHANNEL_X_SHIFT) |
    (SRC_ZERO << A1_SRC0_CHANNEL_Y_SHIFT) |
    (SRC_ZERO << A1_SRC0_CHANNEL_Z_SHIFT) |
    (SRC_ONE << A1_SRC0_CHANNEL_W_SHIFT)),
   0};

void
i915_use_passthrough_shader(struct i915_fragment_shader *fs)
{
   fs->program = (uint32_t *)MALLOC(sizeof(passthrough_program));
   if (fs->program) {
      memcpy(fs->program, passthrough_program, sizeof(passthrough_program));
      fs->program_len = ARRAY_SIZE(passthrough_program);
   }
   fs->num_constants = 0;
}

uint32_t
i915_get_temp(struct i915_fp_compile *p)
{
   int bit = ffs(~p->temp_flag);
   if (!bit) {
      i915_program_error(p, "i915_get_temp: out of temporaries");
      return 0;
   }

   p->temp_flag |= 1 << (bit - 1);
   return bit - 1;
}

void
i915_release_temp(struct i915_fp_compile *p, int reg)
{
   p->temp_flag &= ~(1 << reg);
}

/**
 * Get unpreserved temporary, a temp whose value is not preserved between
 * PS program phases.
 */
uint32_t
i915_get_utemp(struct i915_fp_compile *p)
{
   int bit = ffs(~p->utemp_flag);
   if (!bit) {
      i915_program_error(p, "i915_get_utemp: out of temporaries");
      return 0;
   }

   p->utemp_flag |= 1 << (bit - 1);
   return UREG(REG_TYPE_U, (bit - 1));
}

void
i915_release_utemps(struct i915_fp_compile *p)
{
   p->utemp_flag = ~0x7;
}

uint32_t
i915_emit_decl(struct i915_fp_compile *p, uint32_t type, uint32_t nr,
               uint32_t d0_flags)
{
   uint32_t reg = UREG(type, nr);

   if (type == REG_TYPE_T) {
      if (p->decl_t & (1 << nr))
         return reg;

      p->decl_t |= (1 << nr);
   } else if (type == REG_TYPE_S) {
      if (p->decl_s & (1 << nr))
         return reg;

      p->decl_s |= (1 << nr);
   } else
      return reg;

   if (p->decl < p->declarations + I915_PROGRAM_SIZE) {
      *(p->decl++) = (D0_DCL | D0_DEST(reg) | d0_flags);
      *(p->decl++) = D1_MBZ;
      *(p->decl++) = D2_MBZ;
   } else
      i915_program_error(p, "Out of declarations");

   p->nr_decl_insn++;
   return reg;
}

uint32_t
i915_emit_arith(struct i915_fp_compile *p, uint32_t op, uint32_t dest,
                uint32_t mask, uint32_t saturate, uint32_t src0, uint32_t src1,
                uint32_t src2)
{
   uint32_t c[3];
   uint32_t nr_const = 0;

   assert(GET_UREG_TYPE(dest) != REG_TYPE_CONST);
   dest = UREG(GET_UREG_TYPE(dest), GET_UREG_NR(dest));
   assert(dest);

   if (GET_UREG_TYPE(src0) == REG_TYPE_CONST)
      c[nr_const++] = 0;
   if (GET_UREG_TYPE(src1) == REG_TYPE_CONST)
      c[nr_const++] = 1;
   if (GET_UREG_TYPE(src2) == REG_TYPE_CONST)
      c[nr_const++] = 2;

   /* Recursively call this function to MOV additional const values
    * into temporary registers.  Use utemp registers for this -
    * currently shouldn't be possible to run out, but keep an eye on
    * this.
    */
   if (nr_const > 1) {
      uint32_t s[3], first, i, old_utemp_flag;

      s[0] = src0;
      s[1] = src1;
      s[2] = src2;
      old_utemp_flag = p->utemp_flag;

      first = GET_UREG_NR(s[c[0]]);
      for (i = 1; i < nr_const; i++) {
         if (GET_UREG_NR(s[c[i]]) != first) {
            uint32_t tmp = i915_get_utemp(p);

            i915_emit_arith(p, A0_MOV, tmp, A0_DEST_CHANNEL_ALL, 0, s[c[i]], 0,
                            0);
            s[c[i]] = tmp;
         }
      }

      src0 = s[0];
      src1 = s[1];
      src2 = s[2];
      p->utemp_flag = old_utemp_flag; /* restore */
   }

   if (p->csr < p->program + I915_PROGRAM_SIZE) {
      *(p->csr++) = (op | A0_DEST(dest) | mask | saturate | A0_SRC0(src0));
      *(p->csr++) = (A1_SRC0(src0) | A1_SRC1(src1));
      *(p->csr++) = (A2_SRC1(src1) | A2_SRC2(src2));
   }

   if (GET_UREG_TYPE(dest) == REG_TYPE_R)
      p->register_phases[GET_UREG_NR(dest)] = p->nr_tex_indirect;

   p->nr_alu_insn++;
   return dest;
}

/**
 * Emit a texture load or texkill instruction.
 * \param dest  the dest i915 register
 * \param destmask  the dest register writemask
 * \param sampler  the i915 sampler register
 * \param coord  the i915 source texcoord operand
 * \param opcode  the instruction opcode
 */
uint32_t
i915_emit_texld(struct i915_fp_compile *p, uint32_t dest, uint32_t destmask,
                uint32_t sampler, uint32_t coord, uint32_t opcode,
                uint32_t coord_mask)
{
   const uint32_t k = UREG(GET_UREG_TYPE(coord), GET_UREG_NR(coord));

   uint32_t coord_used = 0xf << UREG_CHANNEL_X_SHIFT;
   if (coord_mask & TGSI_WRITEMASK_Y)
      coord_used |= 0xf << UREG_CHANNEL_Y_SHIFT;
   if (coord_mask & TGSI_WRITEMASK_Z)
      coord_used |= 0xf << UREG_CHANNEL_Z_SHIFT;
   if (coord_mask & TGSI_WRITEMASK_W)
      coord_used |= 0xf << UREG_CHANNEL_W_SHIFT;

   if ((coord & coord_used) != (k & coord_used) ||
       GET_UREG_TYPE(coord) == REG_TYPE_CONST) {
      /* texcoord is swizzled or negated.  Need a temporary to hold it.
       * Use a utemp so it doesn't create a tex indirect phase boundary.
       */
      uint32_t tempReg = i915_get_utemp(p);

      i915_emit_arith(p, A0_MOV, tempReg,
                      A0_DEST_CHANNEL_ALL, /* dest reg, writemask */
                      0,                   /* saturate */
                      coord, 0, 0);        /* src0, src1, src2 */

      /* new src texcoord is tempReg */
      coord = tempReg;
   }

   /* Don't worry about saturate as we only support
    */
   if (destmask != A0_DEST_CHANNEL_ALL) {
      /* if not writing to XYZW... */
      uint32_t tmp = i915_get_utemp(p);
      i915_emit_texld(p, tmp, A0_DEST_CHANNEL_ALL, sampler, coord, opcode,
                      coord_mask);
      i915_emit_arith(p, A0_MOV, dest, destmask, 0, tmp, 0, 0);
      /* XXX release utemp here? */
   } else {
      assert(GET_UREG_TYPE(dest) != REG_TYPE_CONST);
      assert(dest == UREG(GET_UREG_TYPE(dest), GET_UREG_NR(dest)));

      /* Output register being oC or oD defines a phase boundary */
      if (GET_UREG_TYPE(dest) == REG_TYPE_OC ||
          GET_UREG_TYPE(dest) == REG_TYPE_OD)
         p->nr_tex_indirect++;

      /* Reading from an r# register whose contents depend on output of the
       * current phase defines a phase boundary.  Prefer just bumping the
       * phase count (free), but if we'd exceed the HW limit, copy to a
       * utemp instead (costs 1 ALU instruction).
       */
      if (GET_UREG_TYPE(coord) == REG_TYPE_R &&
          p->register_phases[GET_UREG_NR(coord)] == p->nr_tex_indirect) {
         if (p->nr_tex_indirect + 1 < I915_MAX_TEX_INDIRECT) {
            p->nr_tex_indirect++;
         } else {
            uint32_t tmp = i915_get_utemp(p);
            i915_emit_arith(p, A0_MOV, tmp, A0_DEST_CHANNEL_ALL, 0,
                            coord, 0, 0);
            coord = tmp;
         }
      }

      if (p->csr < p->program + I915_PROGRAM_SIZE) {
         *(p->csr++) = (opcode | T0_DEST(dest) | T0_SAMPLER(sampler));

         *(p->csr++) = T1_ADDRESS_REG(coord);
         *(p->csr++) = T2_MBZ;
      }

      if (GET_UREG_TYPE(dest) == REG_TYPE_R)
         p->register_phases[GET_UREG_NR(dest)] = p->nr_tex_indirect;

      p->nr_tex_insn++;
   }

   return dest;
}

static uint32_t
i915_try_const1f_in_reg(struct i915_fp_compile *p, float c0, unsigned reg)
{
   struct i915_fragment_shader *ifs = p->shader;

   for (unsigned idx = 0; idx < 4; idx++) {
      if (ifs->constant_flags[reg] & I915_CONSTFLAG_USER_CH(idx))
         continue;
      if (!(ifs->constant_flags[reg] & I915_CONSTFLAG_IMM(idx)) ||
          ifs->constants[reg][idx] == c0) {
         ifs->constants[reg][idx] = c0;
         ifs->constant_flags[reg] |= I915_CONSTFLAG_IMM(idx);
         if (reg + 1 > ifs->num_constants)
            ifs->num_constants = reg + 1;
         return swizzle(UREG(REG_TYPE_CONST, reg), idx, ZERO, ZERO, ONE);
      }
   }
   return UREG_BAD;
}

static uint32_t
i915_try_emit_const1f(struct i915_fp_compile *p, float c0, int preferred_reg)
{
   if (preferred_reg >= 0) {
      uint32_t r = i915_try_const1f_in_reg(p, c0, preferred_reg);
      if (r != UREG_BAD)
         return r;
   }

   for (unsigned reg = 0; reg < I915_MAX_CONSTANT; reg++) {
      uint32_t r = i915_try_const1f_in_reg(p, c0, reg);
      if (r != UREG_BAD)
         return r;
   }

   i915_program_error(p, "i915_emit_const1f: out of constants");
   return 0;
}

uint32_t
i915_emit_const1f(struct i915_fp_compile *p, float c0)
{
   if (c0 == 0.0)
      return swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO);
   if (c0 == 1.0)
      return swizzle(UREG(REG_TYPE_R, 0), ONE, ONE, ONE, ONE);
   if (c0 == -1.0)
      return negate(swizzle(UREG(REG_TYPE_R, 0), ONE, ONE, ONE, ONE),
                    1, 1, 1, 1);

   return i915_try_emit_const1f(p, c0, -1);
}

uint32_t
i915_emit_const1f_prefer(struct i915_fp_compile *p, float c0,
                         int preferred_reg)
{
   if (c0 == 0.0)
      return swizzle(UREG(REG_TYPE_R, 0), ZERO, ZERO, ZERO, ZERO);
   if (c0 == 1.0)
      return swizzle(UREG(REG_TYPE_R, 0), ONE, ONE, ONE, ONE);
   if (c0 == -1.0)
      return negate(swizzle(UREG(REG_TYPE_R, 0), ONE, ONE, ONE, ONE),
                    1, 1, 1, 1);

   return i915_try_emit_const1f(p, c0, preferred_reg);
}

uint32_t
i915_emit_const2f(struct i915_fp_compile *p, float c0, float c1)
{
   struct i915_fragment_shader *ifs = p->shader;
   unsigned reg, idx;

   if (c0 == 0.0)
      return swizzle(i915_emit_const1f(p, c1), ZERO, X, Z, W);
   if (c0 == 1.0)
      return swizzle(i915_emit_const1f(p, c1), ONE, X, Z, W);

   if (c1 == 0.0)
      return swizzle(i915_emit_const1f(p, c0), X, ZERO, Z, W);
   if (c1 == 1.0)
      return swizzle(i915_emit_const1f(p, c0), X, ONE, Z, W);

   // XXX emit swizzle here for 0, 1, -1 and any combination thereof
   // we can use swizzle + neg for that
   for (reg = 0; reg < I915_MAX_CONSTANT; reg++) {
      uint8_t occupied = (ifs->constant_flags[reg] & 0xf) |
                         (ifs->constant_flags[reg] >> 4);
      if (occupied == 0xf)
         continue;
      for (idx = 0; idx < 3; idx++) {
         if (!(occupied & (3 << idx))) {
            ifs->constants[reg][idx + 0] = c0;
            ifs->constants[reg][idx + 1] = c1;
            ifs->constant_flags[reg] |= (3 << idx); /* immediate bits */
            if (reg + 1 > ifs->num_constants)
               ifs->num_constants = reg + 1;
            return swizzle(UREG(REG_TYPE_CONST, reg), idx, idx + 1, ZERO, ONE);
         }
      }
   }

   i915_program_error(p, "i915_emit_const2f: out of constants");
   return 0;
}

uint32_t
i915_emit_const4f(struct i915_fp_compile *p, float c0, float c1, float c2,
                  float c3)
{
   struct i915_fragment_shader *ifs = p->shader;
   unsigned reg;

   // XXX emit swizzle here for 0, 1, -1 and any combination thereof
   // we can use swizzle + neg for that
   for (reg = 0; reg < I915_MAX_CONSTANT; reg++) {
      if ((ifs->constant_flags[reg] & 0x0f) == 0x0f &&
          ifs->constants[reg][0] == c0 && ifs->constants[reg][1] == c1 &&
          ifs->constants[reg][2] == c2 && ifs->constants[reg][3] == c3) {
         return UREG(REG_TYPE_CONST, reg);
      } else if (ifs->constant_flags[reg] == 0) {

         ifs->constants[reg][0] = c0;
         ifs->constants[reg][1] = c1;
         ifs->constants[reg][2] = c2;
         ifs->constants[reg][3] = c3;
         ifs->constant_flags[reg] = 0x0f;
         if (reg + 1 > ifs->num_constants)
            ifs->num_constants = reg + 1;
         return UREG(REG_TYPE_CONST, reg);
      }
   }

   i915_program_error(p, "i915_emit_const4f: out of constants");
   return 0;
}

uint32_t
i915_emit_const4fv(struct i915_fp_compile *p, const float *c)
{
   return i915_emit_const4f(p, c[0], c[1], c[2], c[3]);
}
