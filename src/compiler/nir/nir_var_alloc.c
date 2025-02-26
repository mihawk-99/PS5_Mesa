/*
 * Copyright © 2025 Valve Corporation
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
 *
 */

/* This file implements helpers which can be used to allocate variables and
 * minimize space used by re-using locations when possible. It works similar
 * to graph coloring register allocation.
 *
 * It has a few limitations, which cause it to fallback to path which doesn't
 * alias variables:
 * - deref casts
 * - functions
 * - zero_initialize_shared_memory=true
 *
 * Their usage is pretty straightforward:
 * - A container for the unallocated variables sharing an address space is
 *   created with nir_var_alloc_setup().
 * - Each unallocated variable can be added with nir_var_alloc_add().
 * - "nir_variable::driver_location" for each added variable is then given a
 *   location using nir_var_alloc_finish(). Locations are allocated starting
 *   from the "start" parameter.
 *
 * Implementation notes
 * ====================
 *
 * It first notes down each use of a variable and whether they might be used
 * later in mark_var_uses(). This is in the form of:
 * - nir_instr::pass_flags, with a bit set for each variable use, and another
 *   to indicate whether it's not used later.
 * - "var_in" (returned by the function), a per-block set of which variables
 *   are used in the block or a later one.
 *
 * It then continues by building an interference graph in get_interferences().
 * This works by just maintaining a live set while visiting each instruction.
 * At each variable use, it adds an interference between that variable and any
 * other live variables.
 *
 * There is a complication with shared variables, since those can be accessed
 * by other invocations:
 *   layout(local_size_x=2) in;
 *   shared uint array1[2], array2[2];
 *   array1[gl_LocalInvocationID] = 42; //line 0
 *   use(array1[gl_LocalInvocationID]); //line 1
 *   array2[gl_LocalInvocationID^1] = 10; //line 2
 *   //If the other invocation runs lines 0 and 1 after line 2 but before
 *   //line 3, the wrong value will be read.
 *   use(array2[gl_LocalInvocationID^1]); //line 3
 * From a single invocation's perspective, "array1" and "array2" do not
 * interfere. However, this will behave incorrectly if both arrays are
 * assigned the same location. This is resolved by considering two shared
 * variables to be interfering if there is no barrier between their uses.
 *
 * This is done by maintaining a set of shared variables have been accessed
 * since the previous barrier ("since_barrier"). Each time we encounter a
 * variable use in get_interferences(), we also add an interference between
 * the variable and each variable in "since_barrier".
 *
 * After the interference graph is made, alloc_using_interferences() allocates
 * locations using a process similar to greedy graph coloring. This is done by
 * iterating over each variable (sorted from largest to smallest for better
 * quality), and creating a set of ranges of interfering variables which are
 * already allocated. The smallest gap is then chosen for the variable.
 */

#include "util/u_dynarray.h"
#include "nir.h"
#include "nir_worklist.h"

struct variable_info {
   nir_variable *var;
   struct set *interferences;
   uint32_t size;
   uint32_t alignment;
};

static bool
update_bitset(BITSET_WORD *dst, BITSET_WORD *src0, BITSET_WORD *src1, unsigned words)
{
   bool changed = false;
   for (unsigned i = 0; i < words; i++) {
      BITSET_WORD src = src0[i] & src1[i];
      changed |= src & ~dst[i];
      dst[i] |= src;
   }
   return changed;
}

static bool
check_var_use(struct nir_var_alloc_state *state, BITSET_WORD *vars, nir_intrinsic_instr *intrin,
              unsigned src)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[src]);
   nir_variable *var = nir_deref_instr_get_variable(deref);

   if (!var && nir_deref_mode_may_be(deref, state->modes))
      return false;

   if (!var || !(var->data.mode & state->modes) || !_mesa_hash_table_search(state->vars, var))
      return true;

   intrin->instr.pass_flags |= 1 << src;
   if (!BITSET_TEST(vars, var->pass_flags))
      intrin->instr.pass_flags |= 1 << (4 + src);

   BITSET_SET(vars, var->pass_flags);

   return true;
}

/* This marks the uses of variables in pass_flags and returns per-block sets of variables which are
 * used in a later block.
 */
static BITSET_WORD *
mark_var_uses(void *mem_ctx, struct nir_var_alloc_state *state, nir_function_impl *impl)
{
   unsigned var_words = BITSET_WORDS(state->vars->entries);
   /* "out" are variables which might be used in a later block. "in" is the same as "out" except it
    * also includes variables used in the block.
    */
   BITSET_WORD *in = rzalloc_array(mem_ctx, BITSET_WORD, impl->num_blocks * var_words);
   BITSET_WORD *out = rzalloc_array(mem_ctx, BITSET_WORD, impl->num_blocks * var_words);

   nir_block_worklist work;
   nir_block_worklist_init(&work, impl->num_blocks, mem_ctx);
   nir_block_worklist_add_all(&work, impl);

   while (!nir_block_worklist_is_empty(&work)) {
      nir_block *block = nir_block_worklist_pop_tail(&work);
      BITSET_WORD *vars = in + block->index * var_words;
      memcpy(vars, out + block->index * var_words, var_words * sizeof(BITSET_WORD));

      nir_foreach_instr_reverse(instr, block) {
         instr->pass_flags = 0;

         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         bool all_uses_simple = true;

         /* When the last use of a variable is both sources of an instruction, prefer marking the
          * second.
          */
         if (intrin->intrinsic == nir_intrinsic_copy_deref)
            all_uses_simple &= check_var_use(state, vars, intrin, 1);

         if (intrin->intrinsic == nir_intrinsic_load_deref ||
             intrin->intrinsic == nir_intrinsic_store_deref ||
             intrin->intrinsic == nir_intrinsic_deref_atomic ||
             intrin->intrinsic == nir_intrinsic_deref_atomic_swap ||
             intrin->intrinsic == nir_intrinsic_copy_deref) {
            all_uses_simple &= check_var_use(state, vars, intrin, 0);
         }

         /* If the variable is unknown, use the simple allocation method.
          * TODO: we could do better here if we follow bcsels/phis
          */
         if (!all_uses_simple)
            return NULL;
      }

      nir_foreach_pred(pred, block) {
         if (update_bitset(out + pred->index * var_words, vars, vars, var_words))
            nir_block_worklist_push_head(&work, (nir_block *)pred);
      }
   }

   nir_block_worklist_fini(&work);
   ralloc_free(out);

   return in;
}

static bool
is_execution_barrier(nir_instr *instr)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(nir_instr_as_intrinsic(instr)) >= SCOPE_WORKGROUP;
}

static void
add_var_interferences(unsigned var, BITSET_WORD *live, BITSET_WORD *since_barrier,
                      unsigned num_vars, struct variable_info **var_infos)
{
   const struct variable_info *info = var_infos[var];

   /* Two different shared variables need a barrier between their uses for them to not interfere. */
   if (since_barrier && info->var->data.mode == nir_var_mem_shared)
      add_var_interferences(var, since_barrier, NULL, num_vars, var_infos);

   unsigned other;
   BITSET_FOREACH_SET(other, live, num_vars) {
      if (other == var)
         continue;
      _mesa_set_add(info->interferences, var_infos[other]);
      _mesa_set_add(var_infos[other]->interferences, info);
   }
}

static bool
get_interferences(struct nir_var_alloc_state *state, nir_function_impl *impl)
{
   nir_metadata_require(impl, nir_metadata_block_index);

   void *tmp_ctx = ralloc_context(state->vars);
   BITSET_WORD *var_in = mark_var_uses(tmp_ctx, state, impl);
   if (!var_in) {
      ralloc_free(tmp_ctx);
      return false;
   }

   /* live_in are the variables which are live at the start of the block. */
   unsigned num_vars = state->vars->entries;
   unsigned var_words = BITSET_WORDS(num_vars);
   BITSET_WORD *live_in = rzalloc_array(tmp_ctx, BITSET_WORD, impl->num_blocks * var_words);
   BITSET_WORD *live = rzalloc_array(tmp_ctx, BITSET_WORD, var_words);

   /* since_barrier_in are the variables which were live at some point since the previous barrier
    * or the start of the shader. */
   BITSET_WORD *since_barrier_in = NULL;
   BITSET_WORD *since_barrier = NULL;
   if (state->modes & nir_var_mem_shared) {
      since_barrier_in = rzalloc_array(tmp_ctx, BITSET_WORD, impl->num_blocks * var_words);
      since_barrier = rzalloc_array(tmp_ctx, BITSET_WORD, var_words);
   }

   nir_block_worklist work;
   nir_block_worklist_init(&work, impl->num_blocks, tmp_ctx);
   nir_block_worklist_add_all(&work, impl);

   /* Obtain variable live-in and the latest execution barriers to create a variable
    * interference graph.
    */
   while (!nir_block_worklist_is_empty(&work)) {
      nir_block *block = nir_block_worklist_pop_head(&work);
      memcpy(live, live_in + block->index * var_words, var_words * sizeof(BITSET_WORD));
      if (since_barrier)
         memcpy(since_barrier, since_barrier_in + block->index * var_words, var_words * sizeof(BITSET_WORD));

      nir_foreach_instr(instr, block) {
         u_foreach_bit(i, instr->pass_flags & 0xf) {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            unsigned var = nir_intrinsic_get_var(intrin, i)->pass_flags;
            struct variable_info **var_infos = state->var_list.data;

            /* Add variables to the live set and update the interference graph. */
            add_var_interferences(var, live, since_barrier, num_vars, var_infos);

            BITSET_SET(live, var);
            if (since_barrier && var_infos[var]->var->data.mode & nir_var_mem_shared)
               BITSET_SET(since_barrier, var);

            if (instr->pass_flags & BITFIELD_BIT(4 + i))
               BITSET_CLEAR(live, var); /* Remove from live set if it's the last use. */
         }

         if (is_execution_barrier(instr))
            memset(since_barrier, 0, var_words * sizeof(BITSET_WORD));
      }

      /* Copy live-out to the successors. */
      for (unsigned i = 0; i < 2; i++) {
         nir_block *succ = block->successors[i];
         if (!succ || succ == impl->end_block)
            continue;

         /* We need to mask "live" by "var_in", because a variable might be live in only 1 successor,
          * preventing the kill bit being set in pass_flags.
          *   use(variable) //this is the last access if cond=false
          *   //variable is live here
          *   if (cond) {
          *      use(variable) //last access: it's removed from the live set here
          *   } else {
          *      //we don't copy the variable to this block's live-in because it's not in the
          *      //"var_in" set
          *   }
          */
         if (update_bitset(live_in + succ->index * var_words, live,
                           var_in + succ->index * var_words, var_words)) {
            nir_block_worklist_push_tail(&work, succ);
         }

         if (since_barrier &&
             update_bitset(since_barrier_in + succ->index * var_words, since_barrier,
                           since_barrier, var_words)) {
            nir_block_worklist_push_tail(&work, succ);
         }
      }
   }

   ralloc_free(tmp_ctx);

   return true;
}

struct nir_var_alloc_state
nir_var_alloc_setup(void)
{
   struct nir_var_alloc_state state;
   state.vars = _mesa_pointer_hash_table_create(NULL);
   state.modes = 0;
   util_dynarray_init(&state.var_list, state.vars);
   return state;
}

void
nir_var_alloc_add(struct nir_var_alloc_state *state, nir_variable *var, uint32_t size,
                  uint32_t alignment)
{
   state->modes |= var->data.mode;

   var->data.driver_location = UINT32_MAX;
   var->pass_flags = state->vars->entries;

   struct variable_info *info = ralloc(state->vars, struct variable_info);
   info->var = var;
   info->interferences = _mesa_pointer_set_create(state->vars);
   info->size = size;
   info->alignment = alignment;
   _mesa_hash_table_insert(state->vars, var, info);
   util_dynarray_append(&state->var_list, info);
}

static int
compare_var(const void *a, const void *b)
{
   const struct variable_info *av = *(const struct variable_info **)a;
   const struct variable_info *bv = *(const struct variable_info **)b;
   if (av->size != bv->size)
      return av->size < bv->size ? 1 : -1;

   unsigned a_index = av->var->pass_flags;
   unsigned b_index = bv->var->pass_flags;
   return a_index < b_index ? -1 : (a_index > b_index ? 1 : 0);
}

struct interference_range {
   uint32_t start;
   uint32_t end;
};

static int
compare_interference(const void *a, const void *b)
{
   const struct interference_range *ar = a;
   const struct interference_range *br = b;
   return ar->start < br->start ? -1 : (ar->start > br->start ? 1 : 0);
}

static bool
alloc_using_interferences(struct nir_var_alloc_state *state, nir_function_impl *impl,
                          uint32_t *max)
{
   if (!state->var_list.size)
      return true;

   if (!get_interferences(state, impl))
      return false;

   unsigned num_vars = util_dynarray_num_elements(&state->var_list, struct variable_info *);
   struct interference_range *ranges =
      ralloc_array(state->vars, struct interference_range, num_vars + 1);

   /* Start with the largest variables. */
   qsort(state->var_list.data, num_vars, sizeof(struct variable_info *), compare_var);

   uint32_t start = *max;

   util_dynarray_foreach(&state->var_list, struct variable_info *, item) {
      struct variable_info *info = *item;
      nir_variable *var = info->var;

      /* Gather ranges of interfering variables. */
      unsigned num_ranges = 0;
      set_foreach(info->interferences, entry) {
         const struct variable_info *other_info = entry->key;
         const nir_variable *other = other_info->var;
         if (other->data.driver_location == UINT32_MAX || other_info->size == 0)
            continue;

         ranges[num_ranges].start = other->data.driver_location;
         ranges[num_ranges].end = other->data.driver_location + other_info->size;
         num_ranges++;
      }

      /* Sort and then combine ranges to avoid intersections. */
      qsort(ranges, num_ranges, sizeof(struct interference_range), compare_interference);
      unsigned num_combined_ranges = MIN2(num_ranges, 1);
      for (unsigned i = 1; i < num_ranges; i++) {
         struct interference_range *prev = &ranges[num_combined_ranges - 1];
         struct interference_range cur = ranges[i];
         if (cur.start < prev->end)
            prev->end = MAX2(prev->end, cur.end);
         else
            ranges[num_combined_ranges++] = cur;
      }

      if (!num_combined_ranges) {
         /* If nothing interferes with this variable, just choose a location at the start. */
         var->data.driver_location = ALIGN_POT(start, info->alignment);
         *max = MAX2(*max, var->data.driver_location + info->size);
         continue;
      }

      /* Find a best fit */
      uint32_t best_offset = ALIGN_POT(ranges[num_combined_ranges - 1].end, info->alignment);
      uint32_t best_size = UINT32_MAX;
      for (unsigned i = 0; i < num_combined_ranges; i++) {
         unsigned prev_end = i ? ranges[i - 1].end : start;
         prev_end = ALIGN_POT(prev_end, info->alignment);
         /* This might be negative because of alignment. */
         int32_t size = (int32_t)ranges[i].start - (int32_t)prev_end;
         if (size >= info->size && size < best_size) {
            best_offset = prev_end;
            best_size = size;
         }
      }

      var->data.driver_location = best_offset;
      *max = MAX2(*max, var->data.driver_location + info->size);
   }

   ralloc_free(state->vars);

   return true;
}

static void
alloc_simple(struct nir_var_alloc_state *state, uint32_t *max)
{
   util_dynarray_foreach(&state->var_list, struct variable_info *, item) {
      struct variable_info *info = *item;
      nir_variable *var = info->var;
      if (!(var->data.mode & state->modes))
         continue;

      var->data.driver_location = ALIGN_POT(*max, info->alignment);
      *max = MAX2(*max, var->data.driver_location + info->size);
   }
}

uint32_t
nir_var_alloc_finish(struct nir_var_alloc_state *state, nir_shader *shader, uint32_t start)
{
   /* Global variables require more work when there are function calls. Maybe also function local
    * variables too (I'm not 100% sure how those work).
    *
    * Doing alloc_using_interferences() with zero_initialize_shared_memory=true would require
    * careful placement of the zero initialization code.
    */
   nir_function *first_func = nir_foreach_function_with_impl_first(shader);
   if (exec_list_length(&shader->functions) != 1 ||
       (shader->info.zero_initialize_shared_memory && (state->modes & nir_var_mem_shared)) ||
       !alloc_using_interferences(state, first_func->impl, &start)) {
      alloc_simple(state, &start);
   }

   return start;
}
