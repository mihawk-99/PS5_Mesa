// SPDX-License-Identifier: MIT

#include "nak.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"

/* NAK expects the driver to provide these.  This compute shader uses neither. */
const struct nak_constant_offset_info nak_const_offsets_base = { 0 };
const struct nak_constant_offset_info nak_const_offsets_turing_graphics = { 0 };

enum nak_ray_query_candidate_intersection_type {
   NAK_RAY_QUERY_CANDIDATE_INTERSECTION_TRIANGLE = 0,
   NAK_RAY_QUERY_CANDIDATE_INTERSECTION_AABB = 1,
};

enum nak_ray_query_committed_intersection_type {
   NAK_RAY_QUERY_COMMITTED_INTERSECTION_TRIANGLE = 1,
   NAK_RAY_QUERY_COMMITTED_INTERSECTION_GENERATED = 2,
};

enum nak_ray_query_test_action {
   NAK_RAY_QUERY_TEST_IGNORE = 0,
   NAK_RAY_QUERY_TEST_CONFIRM = 1,
   NAK_RAY_QUERY_TEST_GENERATE = 2,
   NAK_RAY_QUERY_TEST_TERMINATE = 3,
};

struct nak_shader_bin *
nak_test_compile_sm120_ray_query(const struct nv_device_info *dev);

struct nak_shader_bin *
nak_test_compile_sm120_ray_query(const struct nv_device_info *dev)
{
   struct nak_compiler *nak = nak_compiler_create(dev);
   if (nak == NULL)
      return NULL;

   glsl_type_singleton_init_or_ref();

   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                     nak_nir_options(nak),
                                     "nak ray query test");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_variable *query =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *query_deref = &nir_build_deref_var(&b, query)->def;

   nir_def *data_addr_vec =
      nir_ldc_nv(&b, 2, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                 .align_mul = 8);
   nir_def *data_addr = nir_pack_64_2x32(&b, data_addr_vec);
   /* Test data contains the TLAS address, ray direction, and candidate action,
    * followed by the first and second proceed results, candidate and committed
    * intersection types, primitive index, t, and the first object-to-world
    * column.
    */
   nir_def *tlas = nir_load_global(&b, 1, 64, data_addr, .align_mul = 8);
   nir_def *direction_z = nir_load_global(
      &b, 1, 32, nir_iadd_imm(&b, data_addr, 8),
      .align_mul = 4);
   nir_def *action = nir_load_global(
      &b, 1, 32, nir_iadd_imm(&b, data_addr, 12),
      .align_mul = 4);

   nir_def *flags = nir_bcsel(
      &b, nir_ieq_imm(&b, action, NAK_RAY_QUERY_TEST_CONFIRM),
      nir_imm_int(&b, SpvRayFlagsNoOpaqueKHRMask),
      nir_imm_int(&b, 0));
   nir_rq_initialize(&b, query_deref, tlas,
                     flags, nir_imm_int(&b, 0xff),
                     nir_imm_vec3(&b, 0.25f, 0.25f, -1.0f),
                     nir_imm_float(&b, 0.0f),
                     nir_vec3(&b, nir_imm_float(&b, 0.0f),
                              nir_imm_float(&b, 0.0f), direction_z),
                     nir_imm_float(&b, 10.0f));

   nir_def *first_proceed = nir_rq_proceed(&b, 1, query_deref);
   nir_def *candidate_type = nir_rq_load(
      &b, 1, 32, query_deref,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = false);

   nir_store_global(&b, nir_b2i32(&b, first_proceed),
                    nir_iadd_imm(&b, data_addr, 16));
   nir_store_global(&b, candidate_type,
                    nir_iadd_imm(&b, data_addr, 20));

   nir_push_if(&b, first_proceed);
   {
      nir_push_if(
         &b, nir_iand(&b,
                      nir_ieq_imm(&b, action, NAK_RAY_QUERY_TEST_CONFIRM),
                      nir_ieq_imm(
                         &b, candidate_type,
                         NAK_RAY_QUERY_CANDIDATE_INTERSECTION_TRIANGLE)));
      nir_rq_confirm_intersection(&b, query_deref);
      nir_pop_if(&b, NULL);

      nir_push_if(
         &b, nir_iand(&b,
                      nir_ieq_imm(&b, action, NAK_RAY_QUERY_TEST_GENERATE),
                      nir_ieq_imm(
                         &b, candidate_type,
                         NAK_RAY_QUERY_CANDIDATE_INTERSECTION_AABB)));
      nir_rq_generate_intersection(&b, query_deref, nir_imm_float(&b, 1.5f));
      nir_pop_if(&b, NULL);

      nir_push_if(&b,
                  nir_ieq_imm(&b, action, NAK_RAY_QUERY_TEST_TERMINATE));
      nir_rq_terminate(&b, query_deref);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   nir_push_if(
      &b, nir_iand(&b, first_proceed,
                   nir_ine_imm(&b, action, NAK_RAY_QUERY_TEST_TERMINATE)));
   nir_def *second_proceed_then = nir_rq_proceed(&b, 1, query_deref);
   nir_push_else(&b, NULL);
   nir_def *second_proceed_else = nir_imm_false(&b);
   nir_pop_if(&b, NULL);
   nir_def *second_proceed =
      nir_if_phi(&b, second_proceed_then, second_proceed_else);
   nir_store_global(&b, nir_b2i32(&b, second_proceed),
                    nir_iadd_imm(&b, data_addr, 24));

   nir_def *type = nir_rq_load(
      &b, 1, 32, query_deref,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   nir_store_global(&b, type, nir_iadd_imm(&b, data_addr, 28));

   nir_push_if(&b, nir_ine_imm(&b, type, 0));
   nir_def *primitive = nir_rq_load(
      &b, 1, 32, query_deref,
      .ray_query_value = nir_ray_query_value_intersection_primitive_index,
      .committed = true);
   nir_def *t = nir_rq_load(
      &b, 1, 32, query_deref,
      .ray_query_value = nir_ray_query_value_intersection_t,
      .committed = true);
   nir_def *object_to_world = nir_rq_load(
      &b, 3, 32, query_deref,
      .ray_query_value = nir_ray_query_value_intersection_object_to_world,
      .committed = true, .column = 0);

   nir_store_global(&b, primitive, nir_iadd_imm(&b, data_addr, 32));
   nir_store_global(&b, t, nir_iadd_imm(&b, data_addr, 36));
   nir_store_global(&b, object_to_world,
                    nir_iadd_imm(&b, data_addr, 40));
   nir_pop_if(&b, NULL);

   nak_preprocess_nir(b.shader, nak);
   nak_lower_nir_before_linking(b.shader, nak);
   b.shader->info.io_lowered = true;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct nak_shader_bin *bin =
      nak_compile_shader(b.shader, false, nak, 0, NULL, false);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   nak_compiler_destroy(nak);
   return bin;
}
