/*
 * Copyright (C) 2025 Google LLC.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors (Google):
 *      Romaric Jodin <rjodin@google.com>
 */

#include "compiler.h"
#include "valhall.h"

static bool
replace_with_collect_v2i16(bi_instr *I, unsigned *use_count, bi_instr **defs)
{
   if (I->op != BI_OPCODE_MKVEC_V2I16)
      return false;

   /* Check if inputs are only used by this instruction */
   bi_foreach_src(I, s) {
      if (I->src[s].type != BI_INDEX_NORMAL || use_count[I->src[s].value] != 1)
         return false;

      bi_instr *producer = defs[I->src[s].value];
      assert(producer);

      if (!va_op_dest_modifier_does_convert(producer->op))
         return false;
   }

   return true;
}

void
bi_opt_remove_mkvec(bi_context *ctx)
{
   unsigned *use_count = calloc(ctx->ssa_alloc, sizeof(unsigned));
   bi_instr **defs = calloc(ctx->ssa_alloc, sizeof(bi_instr *));

   bi_foreach_instr_global(ctx, I) {
      bi_foreach_ssa_src(I, s) {
         use_count[I->src[s].value]++;
      }
      bi_foreach_ssa_dest(I, d) {
         defs[I->dest[d].value] = I;
      }
   }

   bi_foreach_instr_global_safe(ctx, I) {
      if (replace_with_collect_v2i16(I, use_count, defs))
         I->op = BI_OPCODE_COLLECT_V2I16;
   }

   free(use_count);
   free(defs);
}
