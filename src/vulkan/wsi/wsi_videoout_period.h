/* SPDX-License-Identifier: MIT */
#ifndef WSI_VIDEOOUT_PERIOD_H
#define WSI_VIDEOOUT_PERIOD_H

#include <stdbool.h>
#include <stdint.h>

/* A zero period represents a failed wait. The VRR window excludes an
 * apparent fixed 59.94 Hz output but allows a slower idle floor. */
static inline bool
wsi_videoout_keep_high_refresh(bool configured, bool vrr, uint64_t period_ns)
{
   if (!configured || period_ns == 0)
      return false;
   if (vrr)
      return period_ns < UINT64_C(16200000) || period_ns > UINT64_C(17200000);
   return period_ns <= UINT64_C(12500000);
}

#endif
