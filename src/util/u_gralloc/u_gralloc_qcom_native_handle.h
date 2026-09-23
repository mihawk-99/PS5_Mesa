/*
 * Mesa 3-D graphics library
 *
 * Copyright © 2025, Google Inc.
 * SPDX-License-Identifier: MIT
 */

#ifndef U_GRALLOC_QCOM_NATIVE_HANDLE_H
#define U_GRALLOC_QCOM_NATIVE_HANDLE_H

#include <stdbool.h>

#include "u_gralloc_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

bool
u_gralloc_has_supported_qcom_native_handle(struct u_gralloc_buffer_handle *hnd);

void
u_gralloc_qcom_native_handle_apply_modifier(
   struct u_gralloc_buffer_handle *hnd,
   struct u_gralloc_buffer_basic_info *out);

#ifdef __cplusplus
}
#endif

#endif /* U_GRALLOC_QCOM_NATIVE_HANDLE_H */
