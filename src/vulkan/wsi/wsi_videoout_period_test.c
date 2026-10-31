/* SPDX-License-Identifier: MIT */
#include <stdio.h>

#include "wsi_videoout_period.h"

int
main(void)
{
   static const struct {
      bool configured;
      bool vrr;
      uint64_t period;
      bool keep;
   } cases[] = {
      { false, false, 8333333, false },
      { false, true, 20833333, false },
      { true, false, 0, false },
      { true, true, 0, false },
      { true, false, 8333333, true },
      { true, false, 12500000, true },
      { true, false, 12500001, false },
      { true, false, 16683333, false },
      { true, false, 20833333, false },
      { true, false, 25000000, false },
      { true, true, 8333333, true },
      { true, true, 12500000, true },
      { true, true, 16199999, true },
      { true, true, 16200000, false },
      { true, true, 16683333, false },
      { true, true, 17200000, false },
      { true, true, 17200001, true },
      { true, true, 20833333, true },
      { true, true, 25000000, true },
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      if (wsi_videoout_keep_high_refresh(cases[i].configured, cases[i].vrr,
                                        cases[i].period) != cases[i].keep) {
         fprintf(stderr, "period policy case %u failed\n", i);
         return 1;
      }
   }
   return 0;
}
