/* Copyright 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

/* Lower non-uniform UBO and SSBO loads, stores, and atomics to global memory intrinsics and
 * do the bounds checking in the shader. This is used when it's faster than waterfall loops.
 *
 * This could be generalized to lowering any UBO and SSBO access to global as long as buffer
 * addresses and sizes are queryable.
 *
 * It works as follows:
 *
 * ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT is the granularity at which the out-of-bounds condition
 * can change, and is equal to either:
 * - VkPhysicalDeviceRobustness2PropertiesKHR::robustUniformBufferAccessSizeAlignment, or
 * - VkPhysicalDeviceRobustness2PropertiesKHR::robustStorageBufferAccessSizeAlignment
 *
 * If the instruction isn't fully in bounds, the instruction is split into robust_buffer_access-
 * _size_alignment-sized instructions, which are bounds-checked separately. Since load/store
 * vectorization can merge in-bounds and out-of-bounds accesses, we must always do the splitting
 * if the instruction isn't fully in bounds.
 *
 * Loads are lowered to:
 *
 *    if (in_bounds) {
 *       ... = load_global everything
 *    } else {
 *       if (the load can cross a robust access size alignment boundary) {
 *          for (each robust access size alignment subset except the last one) {
 *             if (subset in bounds)
 *                ... = load_global subset
 *             else
 *                ... = zero
 *          }
 *       }
 *    }
 *
 * The last subset is always out-of-bounds if the original load is at least partially
 * out-of-bounds.
 *
 * Stores are lowered to:
 *
 *    if (in_bounds) {
 *       store_global everything
 *    } else {
 *       if (the store can cross a robust access size alignment boundary) {
 *          for (each robust access size alignment subset except the last one) {
 *             if (subset in bounds)
 *                store_global subset
 *          }
 *       }
 *    }
 *
 * Atomics are lowered to:
 *
 *    if (in_bounds)
 *       ... = atomic_global
 *    else
 *       ... = zero
 *
 * The performance of the else statement shouldn't matter because out-of-bounds cases are very
 * unlikely. The for loop is generated unrolled.
 */

#include "ac_nir.h"
#include "nir_builder.h"

#define ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT  4 /* RADV constant */

static nir_def *
get_buffer_address(nir_builder *b, nir_def *desc)
{
   nir_def *addr_lo = nir_channel(b, desc, 0);
   nir_def *addr_hi = nir_i2i32(b, nir_u2u16(b, nir_channel(b, desc, 1))); /* sign-extend 16 bits */

   return nir_pack_64_2x32_split(b, addr_lo, addr_hi);
}

static nir_def *
get_ubo_address(nir_builder *b, nir_def *resource, enum gl_access_qualifier access)
{
   nir_def *desc = nir_ubo_descriptor_amd(b, resource, .access = access);
   return get_buffer_address(b, desc);
}

static nir_def *
get_ssbo_address(nir_builder *b, nir_def *resource, enum gl_access_qualifier access)
{
   nir_def *desc = nir_ssbo_descriptor_amd(b, resource, .access = access);
   return get_buffer_address(b, desc);
}

static bool
lower_non_uniform_ubo_ssbo(nir_builder *b, nir_intrinsic_instr *intr, void *opaque)
{
   if (nir_intrinsic_has_access(intr) &&
       !(nir_intrinsic_access(intr) & ACCESS_NON_UNIFORM))
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo: {
      bool ubo = intr->intrinsic == nir_intrinsic_load_ubo;
      nir_def *addr, *size;

      if (ubo) {
         addr = get_ubo_address(b, intr->src[0].ssa, nir_intrinsic_access(intr));
         size = nir_get_ubo_size(b, 32, intr->src[0].ssa,
                                 .access = nir_intrinsic_access(intr));
      } else {
         addr = get_ssbo_address(b, intr->src[0].ssa, nir_intrinsic_access(intr));
         size = nir_get_ssbo_size(b, 32, intr->src[0].ssa,
                                  .access = nir_intrinsic_access(intr));
      }
      b->shader->info.uses_resource_info_query = true;

      nir_def *offset = intr->src[1].ssa;
      const unsigned payload_size = intr->def.num_components * intr->def.bit_size / 8;
      enum gl_access_qualifier access = (nir_intrinsic_access(intr) & ~ACCESS_NON_UNIFORM) |
                                        (ubo ? ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE |
                                               ACCESS_RESTRICT : 0);
      nir_def *result = NULL, *result_else = NULL;

      /* Use a single load if the whole load is in bounds. */
      nir_if *if_inbounds = nir_push_if(b, nir_uge(b, size, nir_iadd_imm_nuw(b, offset, payload_size)));
      {
         result = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                                  nir_iadd_nuw(b, addr, nir_u2u64(b, offset)),
                                  .access = access,
                                  .align_mul = nir_intrinsic_align_mul(intr),
                                  .align_offset = nir_intrinsic_align_offset(intr));
      }

      const unsigned alignment = nir_intrinsic_align(intr);
      const unsigned split_load_size = MIN2(ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT, alignment);

      /* If the load can cross a robust access size alignment boundary, split it into
       * loads whose size is equal to ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT.
       *
       * The reason is that ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT-sized access can either be fully
       * in bounds or fully out of bounds.
       */
      if (payload_size > split_load_size) {
         nir_push_else(b, if_inbounds);
         {
            unsigned num_loads = DIV_ROUND_UP(payload_size, split_load_size);
            nir_def **split_results = alloca(sizeof(nir_def*) * num_loads);
            nir_def *split_imm_zero = nir_imm_zero(b, 1, split_load_size * 8);

            /* Note that since the original load isn't fully in bounds here, it implies that
             * the last ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT segment must be out of bounds, so we
             * can just set its result to 0 unconditionally.
             */
            split_results[num_loads - 1] = split_imm_zero;

            /* Generate remaining loads with bounds checking. */
            for (unsigned i = 0; i < num_loads - 1; i++) {
               nir_if *if_split_inbounds =
                  nir_push_if(b, nir_uge(b, size, nir_iadd_imm_nuw(b, offset, split_load_size)));
               {
                  split_results[i] =
                     nir_load_global(b, 1, split_load_size * 8,
                                     nir_iadd_nuw(b, addr, nir_u2u64(b, offset)),
                                     .access = access,
                                     .align_mul = split_load_size,
                                     .align_offset = 0);
               }
               nir_pop_if(b, if_split_inbounds);
               /* Set 0 if bounds checking failed. */
               split_results[i] = nir_if_phi(b, split_results[i], split_imm_zero);

               /* Advance the offset. */
               offset = nir_iadd_imm_nuw(b, offset, split_load_size);
            }

            /* Gather, bitcast, and trim the split results. */
            result_else = nir_vec(b, split_results, num_loads);
            result_else = nir_bitcast_vector(b, result_else, intr->def.bit_size);
            result_else = nir_trim_vector(b, result_else, intr->def.num_components);
         }
      } else {
         /* If load splitting isn't needed, the only option is that the load must be wholly
          * out of bounds.
          */
         result_else = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
      }

      nir_pop_if(b, if_inbounds);
      result = nir_if_phi(b, result, result_else);

      nir_def_replace(&intr->def, result);
      return true;
   }

   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      bool store = intr->intrinsic == nir_intrinsic_store_ssbo;
      nir_def *value = intr->src[store ? 0 : 2].ssa;
      nir_def *value2 = intr->intrinsic == nir_intrinsic_ssbo_atomic_swap ? intr->src[3].ssa : NULL;
      nir_def *addr = get_ssbo_address(b, intr->src[store ? 1 : 0].ssa, nir_intrinsic_access(intr));
      nir_def *size = nir_get_ssbo_size(b, 32, intr->src[store ? 1 : 0].ssa,
                                        .access = nir_intrinsic_access(intr));
      b->shader->info.uses_resource_info_query = true;
      nir_def *offset = intr->src[store ? 2 : 1].ssa;
      unsigned payload_size = value->num_components * value->bit_size / 8;
      nir_def *zero = store ? NULL : nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);

      /* If any component is out of bounds, the whole instruction is out of bounds.
       * (only relevant for stores)
       */
      enum gl_access_qualifier access = nir_intrinsic_access(intr) & ~ACCESS_NON_UNIFORM;
      nir_def *in_bounds = nir_uge(b, size, nir_iadd_imm_nuw(b, offset, payload_size));
      nir_def *result = NULL;

      nir_if *if_inbounds = nir_push_if(b, in_bounds);
      {
         if (store) {
            nir_store_global(b, value, nir_iadd_nuw(b, addr, nir_u2u64(b, offset)),
                             .write_mask = nir_intrinsic_write_mask(intr),
                             .access = access,
                             .align_mul = nir_intrinsic_align_mul(intr),
                             .align_offset = nir_intrinsic_align_offset(intr));
         } else if (value2) {
            result = nir_global_atomic_swap(b, intr->def.bit_size,
                                            nir_iadd_nuw(b, addr, nir_u2u64(b, offset)),
                                            value, value2,
                                            /* there is no access field */
                                            .atomic_op = nir_intrinsic_atomic_op(intr));
         } else {
            result = nir_global_atomic(b, intr->def.bit_size,
                                       nir_iadd_nuw(b, addr, nir_u2u64(b, offset)), value,
                                       /* there is no access field */
                                       .atomic_op = nir_intrinsic_atomic_op(intr));
         }
      }

      /* If the store can cross a robust access size alignment boundary, split it into
       * stores whose size is equal to ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT.
       *
       * The reason is that ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT-sized access can either be fully
       * in bounds or fully out of bounds.
       *
       * Atomics are never partially out of bounds.
       */
      if (store) {
         const unsigned alignment = nir_intrinsic_align(intr);
         const unsigned split_store_size = MIN2(ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT, alignment);

         if (payload_size > split_store_size) {
            nir_push_else(b, if_inbounds);
            {
               unsigned num_stores = DIV_ROUND_UP(payload_size, split_store_size);
               assert(num_stores * split_store_size * 8 % value->bit_size == 0);
               /* Bitcast the store value to the element type of the split store. */
               nir_def *vec_value =
                  nir_pad_vector(b, value, num_stores * split_store_size * 8 / value->bit_size);
               vec_value = nir_bitcast_vector(b, value, split_store_size * 8);

               /* Note that since the original store isn't fully in bounds here, it implies that
                * the last ROBUST_BUFFER_ACCESS_SIZE_ALIGNMENT segment must be out of bounds, so we
                * can just ignore it.
                */
               for (unsigned i = 0; i < num_stores - 1; i++) {
                  nir_if *if_split_inbounds =
                     nir_push_if(b, nir_uge(b, size, nir_iadd_imm_nuw(b, offset, split_store_size)));
                  {
                     nir_store_global(b, nir_channel(b, vec_value, i),
                                      nir_iadd_nuw(b, addr, nir_u2u64(b, offset)),
                                      .access = access,
                                      .align_mul = split_store_size,
                                      .align_offset = 0);
                  }
                  nir_pop_if(b, if_split_inbounds);

                  /* Advance the offset. */
                  offset = nir_iadd_imm_nuw(b, offset, split_store_size);
               }
            }
         }
      }

      nir_pop_if(b, if_inbounds);

      /* Only atomics with uses should insert the phi. */
      if (result && !nir_def_is_unused(&intr->def)) {
         result = nir_if_phi(b, result, zero);
         nir_def_replace(&intr->def, result);
      } else {
         nir_instr_remove(&intr->instr);
      }
      return true;
   }

   default:
      return false;
   }
}

bool
ac_nir_lower_non_uniform_ubo_ssbo_to_global(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_non_uniform_ubo_ssbo,
                                     nir_metadata_none, NULL);
}
