/* SPDX-License-Identifier: MIT
 * Copyright 2026 Valve Corporation
 */

#include "nir_test.h"

class nir_opt_intrinsics_test : public nir_test {
protected:
   nir_opt_intrinsics_test()
      : nir_test("nir_opt_intrinsics_test", MESA_SHADER_FRAGMENT)
   {
   }
};

TEST_F(nir_opt_intrinsics_test, factor_bcsel_load_input)
{
   nir_def *cond = nir_load_front_face(b, 1); /* just a random bool */
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *x = nir_load_input(b, 1, 32, zero);
   nir_def *y = nir_load_input(b, 1, 32, zero);

   nir_io_semantics sem = {0};
   sem.num_slots = 1;
   sem.location = VARYING_SLOT_VAR3;
   nir_intrinsic_set_io_semantics(nir_def_as_intrinsic(x), sem);

   sem.location = VARYING_SLOT_VAR1;
   nir_intrinsic_set_io_semantics(nir_def_as_intrinsic(y), sem);

   nir_intrinsic_instr *use = nir_use(b, nir_bcsel(b, cond, x, y));
   nir_validate_shader(b->shader, "before input load factoring");

   const nir_opt_intrinsics_options opts = {
      .factor_bcsel_load_input = true,
      .allow_bcsel_load_input_divergent_offset_src = true,
   };
   ASSERT_TRUE(nir_opt_intrinsics(b->shader, &opts));
   nir_validate_shader(b->shader, "after input load factoring");

   nir_opt_constant_folding(b->shader);
   nir_opt_dce(b->shader);

   nir_intrinsic_instr *load = nir_src_as_intrinsic(use->src[0]);
   ASSERT_NE(load, nullptr);
   ASSERT_EQ(load->intrinsic, nir_intrinsic_load_input);
   EXPECT_EQ(nir_intrinsic_io_semantics(load).location, VARYING_SLOT_VAR1);
   EXPECT_EQ(nir_intrinsic_io_semantics(load).num_slots, 3);

   nir_alu_instr *bcsel = nir_src_as_alu(load->src[0]);
   ASSERT_NE(bcsel, nullptr);
   ASSERT_EQ(bcsel->op, nir_op_bcsel);
   EXPECT_EQ(bcsel->src[0].src.ssa, cond);
   ASSERT_TRUE(nir_src_is_const(bcsel->src[1].src));
   ASSERT_TRUE(nir_src_is_const(bcsel->src[2].src));
   EXPECT_EQ(nir_src_as_uint(bcsel->src[1].src), 2);
   EXPECT_EQ(nir_src_as_uint(bcsel->src[2].src), 0);

   unsigned num_loads = 0;
   nir_foreach_block(block, b->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_input)
            num_loads++;
      }
   }
   EXPECT_EQ(num_loads, 1);
   nir_validate_shader(b->shader, "after cleanup");
}
