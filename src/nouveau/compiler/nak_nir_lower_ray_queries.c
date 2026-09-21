/* SPDX-License-Identifier: MIT */

#include "nak_private.h"
#include "nir_builder.h"

#include "compiler/spirv/spirv.h"
#include "util/hash_table.h"

/*
 * Lowers ray-query intrinsics to the SM120 hardware ray-query transaction
 * plus a software-managed query object.
 *
 * Only rayQueryProceedEXT issues hardware traversal (one or more
 * transactions).  Initialization, termination, candidate confirmation,
 * intersection generation, and all getters are ordinary software over the
 * query object.
 *
 * The query state uses a struct-typed variable that nir_split_struct_vars can
 * turn into registers.  Getters reload object-space rays and instance records
 * from the TLAS instance tables instead of retaining them in the query object.
 */

/*
 * TLAS records used by ray-query getters:
 *
 *    header + 0x20  root node offset
 *    header + 0x30  instance descriptor-page offset
 *    header + 0x38  active instance-leaf offset
 *    header + 0x70  input instance count
 *
 * Descriptor pages contain 16 instances in four parallel arrays:
 *
 *    page + 0x000  { mask/flags, biased leaf index }[16]
 *    page + 0x080  BLAS address[16]
 *    page + 0x100  object-to-world mat3x4[16]
 *    page + 0x400  { SBT offset, custom index }[16]
 *
 * Active instance leaves are 0x40 bytes and contain the inverse mat3x4 at
 * +0x10.
 */
#define NAK_TLAS_ROOT_NODE_OFFSET      0x20
#define NAK_TLAS_DESCRIPTOR_PTR_OFFSET 0x30
#define NAK_TLAS_INSTANCE_LEAF_PTR_OFFSET 0x38
#define NAK_TLAS_INSTANCE_COUNT_OFFSET 0x70

/* TLAS instance descriptor pages: 16 instances per 0x480-byte page */
#define NAK_TLAS_PAGE_SIZE             0x480
#define NAK_TLAS_PAGE_TRANSFORMS       0x100
#define NAK_TLAS_PAGE_METADATA         0x400
#define NAK_TLAS_LEAF_INDEX_BIAS       0x12
#define NAK_TLAS_INSTANCE_LEAF_SIZE    0x40

/* Event descriptor classes (q[0x30] pair A word 1 high bits) */
#define NAK_RQ_EVENT_CLASS_MASK        0xfe000000
#define NAK_RQ_EVENT_CLASS_AABB        0xa0000000
#define NAK_RQ_EVENT_CLASS_TRI_EX      0xa2000000
#define NAK_RQ_EVENT_CLASS_DONE        0xe0000000

/* Values returned by rayQueryGetIntersectionTypeEXT */
#define NAK_RQ_COMMITTED_NONE      0
#define NAK_RQ_COMMITTED_TRIANGLE  1
#define NAK_RQ_COMMITTED_GENERATED 2
#define NAK_RQ_CANDIDATE_TRIANGLE  0
#define NAK_RQ_CANDIDATE_AABB      1

enum nak_rq_field {
   /*
    * Software query-object state:
    *
    *    traversal input     tlas, ray, flags, mask
    *    hardware resume     cont10, cont18, cont19, first_transaction
    *    traversal status    t (current tmax), incomplete
    *    candidate payload   cand_*
    *    committed payload   com_*, committed_type
    *
    * Candidate confirmation copies cand_* to com_* and narrows t.  Calling
    * proceed again reuses the continuation vectors left by the transaction.
    */
   /* TLAS base address */
   nak_rq_field_tlas,
   /* Twelve continuation words as three uvec4s.  Components 0-1 are input
    * pair A, components 2-3 are pair B.  cont10 is also transferred as
    * slot 0x20.
    */
   nak_rq_field_cont10,
   nak_rq_field_cont18,
   nak_rq_field_cont19,
   nak_rq_field_origin,
   nak_rq_field_tmin,
   nak_rq_field_direction,
   /* Committed t; initialized to tmax */
   nak_rq_field_t,
   nak_rq_field_flags,
   nak_rq_field_cull_mask,
   nak_rq_field_committed_type,
   nak_rq_field_candidate_type,
   /* False once traversal has completed or the query was terminated */
   nak_rq_field_incomplete,
   /* Selects the root-input protocol until the first transaction completes */
   nak_rq_field_first_transaction,
   nak_rq_field_candidate_aabb_opaque,
   /* Candidate intersection */
   nak_rq_field_cand_t,
   nak_rq_field_cand_prim,
   nak_rq_field_cand_geom,
   nak_rq_field_cand_inst,
   nak_rq_field_cand_sbt,
   /* Raw payload word; face orientation in bit 31, barycentric u in the
    * low 31 bits
    */
   nak_rq_field_cand_bary_u,
   nak_rq_field_cand_bary_v,
   /* Committed intersection */
   nak_rq_field_com_prim,
   nak_rq_field_com_geom,
   nak_rq_field_com_inst,
   nak_rq_field_com_sbt,
   nak_rq_field_com_bary_u,
   nak_rq_field_com_bary_v,
   nak_rq_field_count,
};

static const glsl_type *
nak_get_ray_query_type(void)
{
   glsl_struct_field fields[nak_rq_field_count];

#define FIELD(field_name, field_type)                    \
   fields[nak_rq_field_##field_name] = (glsl_struct_field){ \
      .type = field_type,                                \
      .name = #field_name,                               \
   }

   FIELD(tlas, glsl_uint64_t_type());
   FIELD(cont10, glsl_uvec4_type());
   FIELD(cont18, glsl_uvec4_type());
   FIELD(cont19, glsl_uvec4_type());
   FIELD(origin, glsl_vec_type(3));
   FIELD(tmin, glsl_float_type());
   FIELD(direction, glsl_vec_type(3));
   FIELD(t, glsl_float_type());
   FIELD(flags, glsl_uint_type());
   FIELD(cull_mask, glsl_uint_type());
   FIELD(committed_type, glsl_uint_type());
   FIELD(candidate_type, glsl_uint_type());
   FIELD(incomplete, glsl_bool_type());
   FIELD(first_transaction, glsl_bool_type());
   FIELD(candidate_aabb_opaque, glsl_bool_type());
   FIELD(cand_t, glsl_float_type());
   FIELD(cand_prim, glsl_uint_type());
   FIELD(cand_geom, glsl_uint_type());
   FIELD(cand_inst, glsl_uint_type());
   FIELD(cand_sbt, glsl_uint_type());
   FIELD(cand_bary_u, glsl_uint_type());
   FIELD(cand_bary_v, glsl_uint_type());
   FIELD(com_prim, glsl_uint_type());
   FIELD(com_geom, glsl_uint_type());
   FIELD(com_inst, glsl_uint_type());
   FIELD(com_sbt, glsl_uint_type());
   FIELD(com_bary_u, glsl_uint_type());
   FIELD(com_bary_v, glsl_uint_type());

#undef FIELD

   return glsl_struct_type(fields, nak_rq_field_count, "nak_ray_query",
                           false);
}

#define rq_deref(b, deref, field) \
   nir_build_deref_struct(b, deref, nak_rq_field_##field)
#define rq_load(b, deref, field) nir_load_deref(b, rq_deref(b, deref, field))
#define rq_store(b, deref, field, value)                     \
   nir_store_deref(b, rq_deref(b, deref, field), value,     \
                   BITFIELD_MASK((value)->num_components))

static nir_def *
load_global_u32(nir_builder *b, nir_def *addr)
{
   return nir_load_global(b, 1, 32, addr, .align_mul = 4);
}

static nir_def *
load_global_u64(nir_builder *b, nir_def *addr)
{
   return nir_load_global(b, 1, 64, addr, .align_mul = 8);
}

static nir_def *
sign_extend_i8(nir_builder *b, nir_def *x)
{
   return nir_ishr_imm(b, nir_ishl_imm(b, x, 24), 24);
}

static void
lower_rq_initialize(nir_builder *b, nir_intrinsic_instr *intrin,
                    nir_deref_instr *rq)
{
   /*
    * Software query-state initialization:
    *
    *    query.ray = { origin, tmin, direction }
    *    query.t = tmax
    *    query.committed = None
    *    query.incomplete = (tlas != NULL)
    *    query.first_transaction = true
    *    query.continuation = 0
    *
    * For a non-null TLAS, cont10.xy receives the tagged root-node address.
    * The first transaction selects that root; later transactions replace it
    * with the continuation words returned by the transaction outputs.
    */
   nir_def *tlas = intrin->src[1].ssa;
   nir_def *ray_flags = intrin->src[2].ssa;
   nir_def *cull_mask = nir_iand_imm(b, intrin->src[3].ssa, 0xff);
   nir_def *origin = intrin->src[4].ssa;
   nir_def *tmin = intrin->src[5].ssa;
   nir_def *direction = intrin->src[6].ssa;
   nir_def *tmax = intrin->src[7].ssa;

   rq_store(b, rq, tlas, tlas);
   rq_store(b, rq, origin, origin);
   rq_store(b, rq, tmin, tmin);
   rq_store(b, rq, direction, direction);
   rq_store(b, rq, t, tmax);
   rq_store(b, rq, flags, ray_flags);
   rq_store(b, rq, cull_mask, cull_mask);
   rq_store(b, rq, committed_type, nir_imm_int(b, NAK_RQ_COMMITTED_NONE));
   rq_store(b, rq, candidate_type, nir_imm_int(b, NAK_RQ_CANDIDATE_TRIANGLE));
   nir_def *has_tlas = nir_ine_imm(b, tlas, 0);
   rq_store(b, rq, incomplete, has_tlas);
   rq_store(b, rq, first_transaction, nir_imm_true(b));
   rq_store(b, rq, candidate_aabb_opaque, nir_imm_false(b));

   nir_def *zero = nir_imm_int(b, 0);
   rq_store(b, rq, cont10, nir_vec4(b, zero, zero, zero, zero));
   rq_store(b, rq, cont18, nir_vec4(b, zero, zero, zero, zero));
   rq_store(b, rq, cont19, nir_vec4(b, zero, zero, zero, zero));

   /*
    * A null acceleration-structure descriptor initializes an already
    * complete query.  Keep the root load under control flow: selecting
    * between its result and zero would still allow the global load from
    * address 0x20 to be speculated.
    */
   nir_push_if(b, has_tlas);
   {
      /* root = tlas + load_u64(tlas + 0x20), tagged in the high word */
      nir_def *root = nir_iadd(
         b, tlas,
         load_global_u64(
            b, nir_iadd_imm(b, tlas, NAK_TLAS_ROOT_NODE_OFFSET)));
      nir_def *root_lo = nir_unpack_64_2x32_split_x(b, root);
      nir_def *root_hi =
         nir_ior_imm(b, nir_unpack_64_2x32_split_y(b, root), 0x10000000);

      rq_store(b, rq, cont10,
               nir_vec4(b, root_lo, root_hi, zero, zero));
   }
   nir_pop_if(b, NULL);
}

static void
lower_rq_terminate(nir_builder *b, nir_deref_instr *rq)
{
   rq_store(b, rq, incomplete, nir_imm_false(b));
}

static void
copy_candidate_to_committed(nir_builder *b, nir_deref_instr *rq)
{
   rq_store(b, rq, com_prim, rq_load(b, rq, cand_prim));
   rq_store(b, rq, com_geom, rq_load(b, rq, cand_geom));
   rq_store(b, rq, com_inst, rq_load(b, rq, cand_inst));
   rq_store(b, rq, com_sbt, rq_load(b, rq, cand_sbt));
   rq_store(b, rq, com_bary_u, rq_load(b, rq, cand_bary_u));
   rq_store(b, rq, com_bary_v, rq_load(b, rq, cand_bary_v));
}

static void
lower_rq_confirm_intersection(nir_builder *b, nir_deref_instr *rq)
{
   copy_candidate_to_committed(b, rq);
   rq_store(b, rq, t, rq_load(b, rq, cand_t));
   rq_store(b, rq, committed_type,
            nir_imm_int(b, NAK_RQ_COMMITTED_TRIANGLE));
}

static void
lower_rq_generate_intersection(nir_builder *b, nir_intrinsic_instr *intrin,
                               nir_deref_instr *rq)
{
   copy_candidate_to_committed(b, rq);
   rq_store(b, rq, t, intrin->src[1].ssa);
   rq_store(b, rq, committed_type,
            nir_imm_int(b, NAK_RQ_COMMITTED_GENERATED));
}

/*
 * Construct the two control vectors consumed by input slots 0x00 and 0x08.
 * Their variable fields are:
 *
 *    face_and_termination =
 *       (TerminateOnFirstHit ? 0x80 : 0) |
 *       (CullBackFacing      ? 0x10 : 0) |
 *       (CullFrontFacing     ? 0x20 : 0)
 *
 *    opacity =
 *       (ForceOpaque         ? 0x20 : 0) |
 *       (ForceNonOpaque      ? 0x02 : 0) |
 *       (CullOpaque          ? 0x01 : 0) |
 *       (CullNonOpaque       ? 0x10 : 0)
 *
 *    q[00] = {
 *       mask << 24 | 0x00030207 | face_and_termination,
 *       0x50455400 | opacity,
 *       mask << 16,
 *       0
 *    }
 *
 *    q[08] = {
 *       mask << 24 | 0x00030407 | face_and_termination,
 *       0x50400400 | opacity |
 *          (SkipTriangles ? 0x00005000 : 0) |
 *          (SkipAABBs     ? 0x00050000 : 0),
 *       mask << 16,
 *       0
 *    }
 *
 * SkipClosestHitShader has no effect on ray-query traversal and therefore
 * contributes no control bit.
 */

/* Face-culling and termination bits shared by both control vectors. */
static nir_def *
build_face_and_termination_bits(nir_builder *b, nir_def *flags)
{
   return nir_ior(b,
      nir_bcsel(b, nir_test_mask(b, flags,
                                SpvRayFlagsTerminateOnFirstHitKHRMask),
                nir_imm_int(b, 0x80), nir_imm_int(b, 0)),
      nir_ior(b,
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsCullBackFacingTrianglesKHRMask),
                   nir_imm_int(b, 0x10), nir_imm_int(b, 0)),
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsCullFrontFacingTrianglesKHRMask),
                   nir_imm_int(b, 0x20), nir_imm_int(b, 0))));
}

static void
build_control_words(nir_builder *b, nir_def *flags, nir_def *mask,
                    nir_def **slot00, nir_def **slot08)
{
   nir_def *face_and_termination =
      build_face_and_termination_bits(b, flags);

   nir_def *opacity = nir_ior(b,
      nir_ior(b,
         nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsOpaqueKHRMask),
                   nir_imm_int(b, 0x20), nir_imm_int(b, 0)),
         nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsNoOpaqueKHRMask),
                   nir_imm_int(b, 0x02), nir_imm_int(b, 0))),
      nir_ior(b,
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsCullOpaqueKHRMask),
                   nir_imm_int(b, 0x01), nir_imm_int(b, 0)),
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsCullNoOpaqueKHRMask),
                   nir_imm_int(b, 0x10), nir_imm_int(b, 0))));

   nir_def *mask24 = nir_ishl_imm(b, mask, 24);

   nir_def *c0 = nir_ior(b, nir_ior_imm(b, mask24, 0x00030207),
                         face_and_termination);
   nir_def *c1 = nir_ior_imm(b, opacity, 0x50455400);
   nir_def *c2 = nir_ishl_imm(b, mask, 16);
   nir_def *c3 = nir_imm_int(b, 0);
   nir_def *c4 = nir_ior(b, nir_ior_imm(b, mask24, 0x00030407),
                         face_and_termination);
   nir_def *c5 = nir_ior(b, nir_ior_imm(b, opacity, 0x50400400),
      nir_ior(b,
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsSkipTrianglesKHRMask),
                   nir_imm_int(b, 0x00005000), nir_imm_int(b, 0)),
         nir_bcsel(b, nir_test_mask(b, flags,
                                   SpvRayFlagsSkipAABBsKHRMask),
                   nir_imm_int(b, 0x00050000), nir_imm_int(b, 0))));

   *slot00 = nir_vec4(b, c0, c1, c2, c3);
   *slot08 = nir_vec4(b, c4, c5, c2, c3);
}

/* Feed a bundle output back as an input: the pair order is swapped. */
static nir_def *
pair_swap(nir_builder *b, nir_def *v)
{
   return nir_vec4(b, nir_channel(b, v, 2), nir_channel(b, v, 3),
                   nir_channel(b, v, 0), nir_channel(b, v, 1));
}

struct nak_ttu_op_bundle_out {
   nir_def *slot38, *slot3c, *slot3d, *slot32, *slot30;
};

static nir_def *
unpack_bundle_output(nir_builder *b, nir_def *bundle, unsigned output)
{
   nir_def *pair_a =
      nir_unpack_64_2x32(b, nir_channel(b, bundle, output * 2));
   nir_def *pair_b =
      nir_unpack_64_2x32(b, nir_channel(b, bundle, output * 2 + 1));
   return nir_vec4(b, nir_channel(b, pair_a, 0),
                   nir_channel(b, pair_a, 1),
                   nir_channel(b, pair_b, 0),
                   nir_channel(b, pair_b, 1));
}

/*
 * Emit one complete TTU operation bundle.  The first bundle sends the root
 * through slot 0x20.  Resumed bundles send the three continuation
 * groups through slots 0x10, 0x18, and 0x19.  Complementary lane-wise
 * predicates ensure that only one form executes.
 *
 * Input pseudocode:
 *
 *    q[00] = control(flags, mask)
 *    q[04] = { origin.xyz, tmin }
 *    q[05] = { direction.xyz, committed_t }
 *    q[08] = control(flags, mask)
 *    if first:
 *       q[20] = root
 *    else:
 *       q[10], q[18], q[19] = saved continuation
 *    execute
 *
 * The bundle returns three continuation vectors then three event vectors:
 *
 *    q[38], q[3c], q[3d]  continuation state
 *    q[30] = { bary_u, descriptor/bary_v, triangle_ref, flagged_t }
 *    q[31].zw             metadata context address
 *    q[32].zw             packed SBT offset, instance index + 1
 *
 * Outputs and the next bundle's inputs use opposite pair order, hence
 * pair_swap().
 */
static struct nak_ttu_op_bundle_out
emit_ttu_op_bundle(nir_builder *b, nir_deref_instr *rq)
{
   nir_def *flags = rq_load(b, rq, flags);
   nir_def *mask = rq_load(b, rq, cull_mask);

   nir_def *slot00, *slot08;
   build_control_words(b, flags, mask, &slot00, &slot08);

   nir_def *origin = rq_load(b, rq, origin);
   nir_def *tmin = rq_load(b, rq, tmin);
   nir_def *direction = rq_load(b, rq, direction);
   /* The committed t is the current tmax */
   nir_def *tmax = rq_load(b, rq, t);

   nir_def *slot04 = nir_vec4(b, nir_channel(b, origin, 0),
                              nir_channel(b, origin, 1),
                              nir_channel(b, origin, 2), tmin);
   nir_def *slot05 = nir_vec4(b, nir_channel(b, direction, 0),
                              nir_channel(b, direction, 1),
                              nir_channel(b, direction, 2), tmax);

   nir_def *cont10 = rq_load(b, rq, cont10);
   nir_def *cont18 = rq_load(b, rq, cont18);
   nir_def *cont19 = rq_load(b, rq, cont19);
   nir_def *first = rq_load(b, rq, first_transaction);
   nir_def *resumed = nir_inot(b, first);
   nir_def *always = nir_imm_true(b);

   nir_def *predicates =
      nir_vec8(b, always, always, always, always,
               resumed, resumed, resumed, first);
   nir_def *bundle =
      nir_ttu_op_bundle_nv(b, slot00, slot04, slot05, slot08,
                           cont10, cont18, cont19, predicates);
   rq_store(b, rq, first_transaction, nir_imm_false(b));

   return (struct nak_ttu_op_bundle_out) {
      .slot38 = unpack_bundle_output(b, bundle, 0),
      .slot3c = unpack_bundle_output(b, bundle, 1),
      .slot3d = unpack_bundle_output(b, bundle, 2),
      .slot32 = unpack_bundle_output(b, bundle, 3),
      .slot30 = unpack_bundle_output(b, bundle, 5),
   };
}

struct nak_rq_tri_decode {
   nir_def *prim;
   nir_def *geom;
};

/*
 * Decode the primitive and geometry index for an ordinary triangle result.
 *
 * The direct form packs both indices into 28 payload bits:
 *
 *    primitive_bits = 28 - 4 * triangle_ref[31:29]
 *    primitive      = triangle_ref[primitive_bits - 1:0]
 *    geometry       = triangle_ref[27:primitive_bits]
 *
 * Extended forms carry a signed 8-bit primitive index but no geometry index.
 */
static struct nak_rq_tri_decode
decode_triangle_metadata(nir_builder *b, nir_def *ref)
{
   /* In the direct packed form, the top three bits select a
    * 12/16/20/24/28-bit primitive field.
    */
   nir_def *is_direct = nir_ult_imm(b, ref, 0xa0000000);
   nir_def *exponent = nir_ushr_imm(b, ref, 29);
   nir_def *split = nir_iadd_imm(
      b, nir_ineg(b, nir_imul_imm(b, exponent, 4)), 28);
   nir_def *high_mask = nir_ishl(b, nir_imm_int(b, 0xffffffff), split);
   nir_def *direct_prim = nir_iand(b, ref, nir_inot(b, high_mask));
   nir_def *direct_geom =
      nir_ushr(b, nir_iand_imm(b, ref, 0x0fffffff), split);

   return (struct nak_rq_tri_decode) {
      .prim = nir_bcsel(b, is_direct, direct_prim,
                        sign_extend_i8(b, nir_iand_imm(b, ref, 0xff))),
      .geom = nir_bcsel(b, is_direct, direct_geom, nir_imm_int(b, 0)),
   };
}

static nir_def *
load_instance_page_base(nir_builder *b, nir_deref_instr *rq, nir_def *inst,
                        nir_def **page_out, nir_def **slot_out)
{
   nir_def *tlas = rq_load(b, rq, tlas);
   nir_def *descriptor_offset = nir_u2u64(
      b, load_global_u32(
            b, nir_iadd_imm(b, tlas, NAK_TLAS_DESCRIPTOR_PTR_OFFSET)));
   nir_def *page = nir_ushr_imm(b, inst, 4);
   nir_def *slot = nir_iand_imm(b, inst, 15);

   if (page_out != NULL)
      *page_out = page;
   if (slot_out != NULL)
      *slot_out = slot;

   return nir_iadd(
      b, nir_iadd(b, tlas, descriptor_offset),
      nir_u2u64(b, nir_imul_imm(b, page, NAK_TLAS_PAGE_SIZE)));
}

/*
 * Load one of the two transforms associated with an instance.  Returns three
 * vec4 rows.
 *
 * Object-to-world matrices are indexed by the original instance index in the
 * descriptor pages:
 *
 *    page = instance / 16
 *    slot = instance % 16
 *    addr = descriptor_base + page * 0x480 + 0x100 + slot * 48
 *
 * World-to-object matrices are stored in the compact active-leaf array.
 * Descriptor word 1 maps the original instance index to that array, after
 * removing the page-dependent traversal bias:
 *
 *    biased_leaf = descriptor_page[slot].word1
 *    leaf_bias = (page_count - page) * 0x12
 *    leaf_index = biased_leaf - leaf_bias
 *    addr = instance_leaf_base + leaf_index * 0x40 + 0x10
 */
static void
load_instance_transform(nir_builder *b, nir_deref_instr *rq, nir_def *inst,
                        bool inverse, nir_def *rows[3])
{
   nir_def *tlas = rq_load(b, rq, tlas);
   nir_def *page;
   nir_def *slot;
   nir_def *page_base =
      load_instance_page_base(b, rq, inst, &page, &slot);
   nir_def *base;

   if (!inverse) {
      base = nir_iadd(
         b, page_base,
         nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, slot, 48),
                                   NAK_TLAS_PAGE_TRANSFORMS)));
   } else {
      nir_def *count = load_global_u32(
         b, nir_iadd_imm(b, tlas, NAK_TLAS_INSTANCE_COUNT_OFFSET));
      nir_def *pages = nir_ushr_imm(b, nir_iadd_imm(b, count, 15), 4);
      nir_def *biased_leaf =
         load_global_u32(
            b, nir_iadd_imm(
                  b, nir_iadd(
                        b, page_base,
                        nir_u2u64(b, nir_imul_imm(b, slot, 8))),
                  4));
      nir_def *leaf_bias =
         nir_imul_imm(b, nir_isub(b, pages, page),
                      NAK_TLAS_LEAF_INDEX_BIAS);
      nir_def *leaf_index = nir_isub(b, biased_leaf, leaf_bias);
      nir_def *leaf_offset = nir_u2u64(
         b, load_global_u32(
               b, nir_iadd_imm(b, tlas,
                               NAK_TLAS_INSTANCE_LEAF_PTR_OFFSET)));
      nir_def *leaf_base = nir_iadd(b, tlas, leaf_offset);
      nir_def *leaf = nir_iadd(
         b, leaf_base,
         nir_u2u64(
            b, nir_imul_imm(b, leaf_index, NAK_TLAS_INSTANCE_LEAF_SIZE)));
      base = nir_iadd_imm(b, leaf, 0x10);
   }

   for (unsigned r = 0; r < 3; r++)
      rows[r] = nir_load_global(b, 4, 32, nir_iadd_imm(b, base, r * 16),
                                .align_mul = 16);
}

static nir_def *
transform_point(nir_builder *b, nir_def *rows[3], nir_def *p)
{
   nir_def *px = nir_channel(b, p, 0);
   nir_def *py = nir_channel(b, p, 1);
   nir_def *pz = nir_channel(b, p, 2);
   nir_def *out[3];
   for (unsigned r = 0; r < 3; r++) {
      nir_def *m = rows[r];
      out[r] = nir_ffma(b, nir_channel(b, m, 0), px,
               nir_ffma(b, nir_channel(b, m, 1), py,
               nir_ffma(b, nir_channel(b, m, 2), pz,
                        nir_channel(b, m, 3))));
   }
   return nir_vec3(b, out[0], out[1], out[2]);
}

static nir_def *
transform_dir(nir_builder *b, nir_def *rows[3], nir_def *v)
{
   nir_def *vx = nir_channel(b, v, 0);
   nir_def *vy = nir_channel(b, v, 1);
   nir_def *vz = nir_channel(b, v, 2);
   nir_def *out[3];
   for (unsigned r = 0; r < 3; r++) {
      nir_def *m = rows[r];
      out[r] = nir_ffma(b, nir_channel(b, m, 0), vx,
               nir_ffma(b, nir_channel(b, m, 1), vy,
                  nir_fmul(b, nir_channel(b, m, 2), vz)));
   }
   return nir_vec3(b, out[0], out[1], out[2]);
}

/*
 * Decode one transaction's return payload into candidate/committed query
 * state, following the top-level event dispatch.  Sets *surface true when
 * rayQueryProceedEXT should return with a shader-visible candidate, and
 * *reissue true when another hardware transaction must run before the
 * proceed can return.
 */
static void
process_return(nir_builder *b, nir_deref_instr *rq,
               struct nak_ttu_op_bundle_out *out,
               nir_variable *surface_var, nir_variable *reissue_var)
{
   nir_def *desc = nir_channel(b, out->slot30, 1);      /* pair A word 1 */
   nir_def *tri_ref = nir_channel(b, out->slot30, 2);   /* pair B word 0 */
   nir_def *flagged_t = nir_channel(b, out->slot30, 3); /* pair B word 1 */
   nir_def *bary_u = nir_channel(b, out->slot30, 0);    /* pair A word 0 */
   nir_def *bary_v = nir_channel(b, out->slot30, 1);

   nir_def *packed_sbt = nir_channel(b, out->slot32, 2); /* pair B word 0 */
   nir_def *inst_plus1 = nir_channel(b, out->slot32, 3); /* pair B word 1 */
   nir_def *inst = nir_iadd_imm(b, inst_plus1, -1);
   nir_def *sbt = nir_iand_imm(b, packed_sbt, 0x0fffffff);
   nir_def *t = nir_iand_imm(b, flagged_t, 0x7fffffff);

   nir_def *cls = nir_iand_imm(b, desc, NAK_RQ_EVENT_CLASS_MASK);
   /* A 0xffffffff return with an exhausted continuation is a completion
    * sentinel.  Checking the event class alone would reissue misses forever.
    */
   nir_def *continuation_empty =
      nir_ieq_imm(b, nir_channel(b, out->slot38, 3), 0);
   nir_def *empty_sentinel = nir_iand(b,
      nir_ieq_imm(b, desc, 0xffffffff),
      continuation_empty);
   nir_def *is_done = nir_ior(b,
      nir_ieq_imm(b, cls, NAK_RQ_EVENT_CLASS_DONE),
      empty_sentinel);
   nir_def *is_aabb = nir_iand(b,
      nir_ieq_imm(b, cls, NAK_RQ_EVENT_CLASS_AABB),
      nir_ine_imm(b, nir_iand_imm(b, desc, 0x3f), 0));
   nir_def *is_tri = nir_ior(b, nir_ige_imm(b, desc, 0),
      nir_ieq_imm(b, cls, NAK_RQ_EVENT_CLASS_TRI_EX));

   nir_store_var(b, surface_var, nir_imm_false(b), 0x1);
   nir_store_var(b, reissue_var, nir_imm_false(b), 0x1);

   nir_push_if(b, is_tri);
   {
      struct nak_rq_tri_decode dec = decode_triangle_metadata(b, tri_ref);

      rq_store(b, rq, cand_t, t);
      rq_store(b, rq, cand_prim, dec.prim);
      rq_store(b, rq, cand_geom, dec.geom);
      rq_store(b, rq, cand_inst, inst);
      rq_store(b, rq, cand_sbt, sbt);
      rq_store(b, rq, cand_bary_u, bary_u);
      rq_store(b, rq, cand_bary_v, bary_v);
      rq_store(b, rq, candidate_type,
               nir_imm_int(b, NAK_RQ_CANDIDATE_TRIANGLE));

      /* Ordinary triangle events use flagged_t[31] to distinguish a
       * non-opaque candidate from an opaque triangle committed by hardware.
       * Extended triangle events carry the same selector in desc[0].
       */
      nir_def *is_tri_ex =
         nir_ieq_imm(b, cls, NAK_RQ_EVENT_CLASS_TRI_EX);
      nir_def *is_candidate = nir_bcsel(
         b, is_tri_ex,
         nir_ine_imm(b, nir_iand_imm(b, desc, 1), 0),
         nir_ine_imm(b, nir_iand_imm(b, flagged_t, 0x80000000), 0));
      nir_store_var(b, surface_var, is_candidate, 0x1);

      /*
       * A triangle with the candidate bit clear has already been accepted
       * by the RT unit.  It is not surfaced to the shader, but its payload
       * is the query's new committed intersection.  Keep the candidate
       * stores above as a convenient staging area and install the result
       * into the committed half here.
       */
      nir_push_if(b, nir_inot(b, is_candidate));
      {
         copy_candidate_to_committed(b, rq);
         rq_store(b, rq, t, t);
         rq_store(b, rq, committed_type,
                  nir_imm_int(b, NAK_RQ_COMMITTED_TRIANGLE));

         /* Continue traversal after an opaque hit so a nearer non-opaque
          * candidate can still be returned to the shader.
          */
         nir_def *terminate = nir_test_mask(
            b, rq_load(b, rq, flags),
            SpvRayFlagsTerminateOnFirstHitKHRMask);
         nir_store_var(b, reissue_var, nir_inot(b, terminate), 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_push_if(b, is_aabb);
      {
         /*
          * AABB payload decoding
          * ---------------------
          *
          * The transaction does not return primitive and geometry directly.
          * It returns a tagged cursor assembled from fields shared with the
          * triangle payload:
          *
          *    f = flags | mask << 16 | committed_type << 24
          *    a = (f >> 1) & 0x7fff
          *    b = ((packed_sbt >> 29) & (f ^ 1)) | a
          *
          *    cursor.low =
          *       triangle_ref * 8 |
          *       ((((packed_sbt >> 28) & ~a) | f) & 1) |
          *       ((b * 2) & 2)
          *
          *    cursor.high =
          *       ((flagged_t & 0x7fff) << 3) |
          *       (triangle_ref >> 29) |
          *       (control_q08_word0 << 17)
          *
          * cursor_low[2:0] contains selector state rather than address bits;
          * the metadata load therefore clears those bits.  cursor_high keeps
          * both address and iteration state.  The record at the resulting
          * address is:
          *
          *    metadata.low        primitive index
          *    metadata.high[23:0] geometry index
          *    metadata.high[24]   encoded opacity
          */
         nir_def *f = nir_ior(b,
            nir_iand_imm(b, rq_load(b, rq, flags), 0xffff),
            nir_ior(b,
               nir_ishl_imm(b, rq_load(b, rq, cull_mask), 16),
               nir_ishl_imm(b, rq_load(b, rq, committed_type), 24)));
         nir_def *a =
            nir_iand_imm(b, nir_ushr_imm(b, f, 1), 0x7fff);
         nir_def *cursor_b = nir_ior(b,
            nir_iand(b, nir_ushr_imm(b, packed_sbt, 29),
                         nir_ixor(b, f, nir_imm_int(b, 1))),
            a);
         nir_def *cursor_low = nir_ior(b,
            nir_imul_imm(b, tri_ref, 8),
            nir_ior(b,
               nir_iand_imm(b,
                  nir_ior(b,
                     nir_iand(b, nir_ushr_imm(b, packed_sbt, 28),
                                  nir_inot(b, a)),
                     f),
                  1),
               nir_iand_imm(b, nir_imul_imm(b, cursor_b, 2), 2)));
         nir_def *mask24 =
            nir_ishl_imm(b, rq_load(b, rq, cull_mask), 24);
         nir_def *control_08_a0 = nir_ior(b,
            nir_ior_imm(b, mask24, 0x00030407),
            build_face_and_termination_bits(b, rq_load(b, rq, flags)));
         nir_def *cursor_high = nir_ior(b,
            nir_ishl_imm(b, nir_iand_imm(b, flagged_t, 0x7fff), 3),
            nir_ior(b,
               nir_ushr_imm(b, tri_ref, 29),
               nir_ishl_imm(b, control_08_a0, 17)));

         /* Form an aligned address from the cursor.  The low 17 bits of
          * cursor.high are GPU address bits 32..48; the remaining bits are
          * traversal state.
          */
         nir_def *metadata_addr = nir_pack_64_2x32_split(b,
            nir_iand_imm(b, cursor_low, 0xfffffff8),
            nir_iand_imm(b, cursor_high, 0x0001ffff));
         nir_def *metadata = load_global_u64(b, metadata_addr);
         nir_def *metadata_lo = nir_unpack_64_2x32_split_x(b, metadata);
         nir_def *metadata_hi = nir_unpack_64_2x32_split_y(b, metadata);

         /* Effective opacity combines the encoded record bit with the ray
          * controls and cursor position.
          */
         nir_def *next_low = nir_iadd_imm(b, cursor_low, 8);
         nir_def *scaled = nir_imul_imm(b, next_low, 0x20000);
         nir_def *record_flag =
            nir_iand_imm(b, nir_ushr_imm(b, metadata_hi, 21), 8);
         nir_def *selector_x = nir_ior(b,
            nir_iand_imm(b, scaled, 0x20000),
            nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, f, 6), 1),
                       record_flag));
         nir_def *selector_y = nir_ior(b,
            nir_iand_imm(b, scaled, 0x40000),
            nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, f, 6), 2),
                       record_flag));
         nir_def *selector = nir_umin(b, nir_iadd_imm(b, selector_x, 4),
                                      nir_ixor(b, selector_y,
                                               nir_imm_int(b, 8)));
         nir_def *opaque = nir_ieq_imm(b, nir_iand_imm(b,
            nir_ishr_imm(b, sign_extend_i8(b, selector), 2), 1), 0);
         nir_def *flags = rq_load(b, rq, flags);
         nir_def *culled = nir_ior(b,
            nir_iand(b, opaque,
               nir_test_mask(b, flags, SpvRayFlagsCullOpaqueKHRMask)),
            nir_iand(b, nir_inot(b, opaque),
               nir_test_mask(b, flags, SpvRayFlagsCullNoOpaqueKHRMask)));

         rq_store(b, rq, cand_prim, metadata_lo);
         rq_store(b, rq, cand_geom,
                  nir_iand_imm(b, metadata_hi, 0x00ffffff));
         rq_store(b, rq, cand_inst, inst);
         rq_store(b, rq, cand_sbt, sbt);
         rq_store(b, rq, candidate_type,
                  nir_imm_int(b, NAK_RQ_CANDIDATE_AABB));
         rq_store(b, rq, candidate_aabb_opaque, opaque);
         nir_store_var(b, surface_var, nir_inot(b, culled), 0x1);
         nir_store_var(b, reissue_var, culled, 0x1);
      }
      nir_push_else(b, NULL);
      {
         /* Internal continuation classes require another transaction. */
         nir_store_var(b, reissue_var, nir_inot(b, is_done), 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   /* Save the returned continuation unless traversal completed. */
   nir_push_if(b, nir_inot(b, is_done));
   {
      rq_store(b, rq, cont10, pair_swap(b, out->slot38));
      rq_store(b, rq, cont18, pair_swap(b, out->slot3c));
      rq_store(b, rq, cont19, pair_swap(b, out->slot3d));
   }
   nir_pop_if(b, NULL);
}

/*
 * rayQueryProceedEXT: run hardware transactions until one surfaces a
 * shader-visible candidate or the query completes.  Returns true if a
 * candidate is available (the shader will inspect/confirm it and call
 * proceed again), false when traversal is finished.
 */
static nir_def *
lower_rq_proceed(nir_builder *b, nir_deref_instr *rq)
{
   nir_variable *surface =
      nir_local_variable_create(b->impl, glsl_bool_type(), "rq_surface");
   nir_variable *reissue =
      nir_local_variable_create(b->impl, glsl_bool_type(), "rq_reissue");

   nir_store_var(b, surface, nir_imm_false(b), 0x1);
   nir_store_var(b, reissue, nir_imm_false(b), 0x1);

   nir_push_if(b, rq_load(b, rq, incomplete));
   {
      nir_push_loop(b);
      {
         struct nak_ttu_op_bundle_out out = emit_ttu_op_bundle(b, rq);
         process_return(b, rq, &out, surface, reissue);

         /* Stop looping unless the return asked to reissue */
         nir_push_if(b, nir_inot(b, nir_load_var(b, reissue)));
         {
            nir_jump(b, nir_jump_break);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_loop(b, NULL);

      /* If nothing surfaced, traversal is complete */
      nir_push_if(b, nir_inot(b, nir_load_var(b, surface)));
      {
         rq_store(b, rq, incomplete, nir_imm_false(b));
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   return nir_load_var(b, surface);
}

static nir_def *
lower_rq_load(nir_builder *b, nir_intrinsic_instr *intrin, nir_deref_instr *rq)
{
   /*
    * Getter lowering reads scalar hit fields from the software query object
    * and reconstructs values that remain packed or resident in the TLAS:
    *
    *    primitive, geometry, instance, SBT, t
    *       -> candidate or committed query-object fields
    *
    *    barycentrics
    *       -> { cand_bary_u[30:0], cand_bary_v }
    *
    *    front_face
    *       -> sign bit of cand_bary_u is clear
    *
    *    custom_index
    *       -> TLAS descriptor-page metadata[instance].custom_index
    *
    *    object-space ray / matrices
    *       -> load the instance transform and apply or extract its rows
    *
    * The committed selector chooses com_* fields; otherwise the getter reads
    * the candidate generated by the most recent proceed call.
    */
   bool committed = nir_intrinsic_committed(intrin);
   unsigned column = nir_intrinsic_column(intrin);
   nir_ray_query_value value = nir_intrinsic_ray_query_value(intrin);

   nir_def *inst = committed ? rq_load(b, rq, com_inst)
                             : rq_load(b, rq, cand_inst);

   switch (value) {
   case nir_ray_query_value_flags:
      return rq_load(b, rq, flags);
   case nir_ray_query_value_tmin:
      return rq_load(b, rq, tmin);
   case nir_ray_query_value_world_ray_origin:
      return rq_load(b, rq, origin);
   case nir_ray_query_value_world_ray_direction:
      return rq_load(b, rq, direction);

   case nir_ray_query_value_intersection_type: {
      if (committed)
         return rq_load(b, rq, committed_type);
      return rq_load(b, rq, candidate_type);
   }
   case nir_ray_query_value_intersection_t:
      return committed ? rq_load(b, rq, t) : rq_load(b, rq, cand_t);
   case nir_ray_query_value_intersection_primitive_index:
      return committed ? rq_load(b, rq, com_prim) : rq_load(b, rq, cand_prim);
   case nir_ray_query_value_intersection_geometry_index:
      return committed ? rq_load(b, rq, com_geom) : rq_load(b, rq, cand_geom);
   case nir_ray_query_value_intersection_instance_id:
      return inst;
   case nir_ray_query_value_intersection_instance_sbt_index:
      return committed ? rq_load(b, rq, com_sbt) : rq_load(b, rq, cand_sbt);
   case nir_ray_query_value_intersection_instance_custom_index: {
      nir_def *slot;
      nir_def *page_base =
         load_instance_page_base(b, rq, inst, NULL, &slot);
      nir_def *md = nir_iadd(
         b, page_base,
         nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, slot, 8),
                                   NAK_TLAS_PAGE_METADATA + 4)));
      return load_global_u32(b, md);
   }
   case nir_ray_query_value_intersection_barycentrics: {
      nir_def *u = committed ? rq_load(b, rq, com_bary_u)
                             : rq_load(b, rq, cand_bary_u);
      nir_def *v = committed ? rq_load(b, rq, com_bary_v)
                             : rq_load(b, rq, cand_bary_v);
      return nir_vec2(b, nir_iand_imm(b, u, 0x7fffffff), v);
   }
   case nir_ray_query_value_intersection_front_face: {
      nir_def *u = committed ? rq_load(b, rq, com_bary_u)
                             : rq_load(b, rq, cand_bary_u);
      /* Front-facing when the stored word is non-negative */
      return nir_ige_imm(b, u, 0);
   }
   case nir_ray_query_value_intersection_candidate_aabb_opaque:
      return rq_load(b, rq, candidate_aabb_opaque);

   case nir_ray_query_value_intersection_object_ray_origin: {
      nir_def *rows[3];
      load_instance_transform(b, rq, inst, true, rows);
      return transform_point(b, rows, rq_load(b, rq, origin));
   }
   case nir_ray_query_value_intersection_object_ray_direction: {
      nir_def *rows[3];
      load_instance_transform(b, rq, inst, true, rows);
      return transform_dir(b, rows, rq_load(b, rq, direction));
   }
   case nir_ray_query_value_intersection_world_to_object: {
      nir_def *rows[3];
      load_instance_transform(b, rq, inst, true, rows);
      return nir_vec3(b, nir_channel(b, rows[0], column),
                      nir_channel(b, rows[1], column),
                      nir_channel(b, rows[2], column));
   }
   case nir_ray_query_value_intersection_object_to_world: {
      nir_def *rows[3];
      load_instance_transform(b, rq, inst, false, rows);
      return nir_vec3(b, nir_channel(b, rows[0], column),
                      nir_channel(b, rows[1], column),
                      nir_channel(b, rows[2], column));
   }

   default:
      UNREACHABLE("Unsupported ray query value");
   }
}

static nir_deref_instr *
lower_opaque_rq_deref(nir_builder *b, nir_deref_instr *opaque, nir_variable *var)
{
   if (opaque->deref_type != nir_deref_type_array)
      return nir_build_deref_var(b, var);

   nir_deref_instr *outer =
      lower_opaque_rq_deref(b, nir_deref_instr_parent(opaque), var);
   return nir_build_deref_array(b, outer, opaque->arr.index.ssa);
}

static void
lower_ray_query_var(nir_shader *shader, nir_variable *ray_query,
                    struct hash_table *ht)
{
   const glsl_type *type =
      glsl_type_wrap_in_arrays(nak_get_ray_query_type(), ray_query->type);
   nir_variable *var = nir_variable_create(shader, nir_var_shader_temp, type,
      ray_query->name ? ray_query->name : "");
   _mesa_hash_table_insert(ht, ray_query, var);
}

bool
nak_nir_lower_ray_queries(nir_shader *shader)
{
   bool progress = false;
   struct hash_table *query_ht = NULL;

   nir_foreach_variable_in_list(var, &shader->variables) {
      if (!var->data.ray_query)
         continue;
      if (query_ht == NULL)
         query_ht = _mesa_pointer_hash_table_create(NULL);
      lower_ray_query_var(shader, var, query_ht);
      progress = true;
   }

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_variable_in_list(var, &impl->locals) {
         if (!var->data.ray_query)
            continue;
         if (query_ht == NULL)
            query_ht = _mesa_pointer_hash_table_create(NULL);
         lower_ray_query_var(shader, var, query_ht);
         progress = true;
      }

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(intrin->intrinsic))
               continue;

            assert(query_ht != NULL);
            nir_deref_instr *opaque = nir_def_as_deref(intrin->src[0].ssa);
            struct hash_entry *he = _mesa_hash_table_search(query_ht,
               nir_deref_instr_get_variable(opaque));
            assert(he);
            nir_variable *var = he->data;

            b.cursor = nir_before_instr(instr);
            nir_deref_instr *rq = lower_opaque_rq_deref(&b, opaque, var);

            nir_def *new_def = NULL;
            switch (intrin->intrinsic) {
            case nir_intrinsic_rq_initialize:
               lower_rq_initialize(&b, intrin, rq);
               break;
            case nir_intrinsic_rq_terminate:
               lower_rq_terminate(&b, rq);
               break;
            case nir_intrinsic_rq_confirm_intersection:
               lower_rq_confirm_intersection(&b, rq);
               break;
            case nir_intrinsic_rq_generate_intersection:
               lower_rq_generate_intersection(&b, intrin, rq);
               break;
            case nir_intrinsic_rq_proceed:
               new_def = lower_rq_proceed(&b, rq);
               break;
            case nir_intrinsic_rq_load:
               new_def = lower_rq_load(&b, intrin, rq);
               break;
            default:
               UNREACHABLE("Unsupported ray query intrinsic");
            }

            if (new_def)
               nir_def_rewrite_uses(&intrin->def, new_def);
            nir_instr_remove(instr);
            nir_instr_free(instr);
            impl_progress = true;
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_none);
   }

   if (query_ht != NULL)
      ralloc_free(query_ht);

   if (progress) {
      NIR_PASS(_, shader, nir_split_struct_vars, nir_var_shader_temp);
      NIR_PASS(_, shader, nir_lower_global_vars_to_local);
      NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   }

   return progress;
}
