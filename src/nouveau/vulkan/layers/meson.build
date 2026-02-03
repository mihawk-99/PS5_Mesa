# Copyright © 2022 Valve Corporation
# SPDX-License-Identifier: MIT

hud_shaders = [
  'hud.frag',
  'hud.vert',
]

hud_spv = []
foreach s : hud_shaders
  _name = f'@s@.spv.h'
  hud_spv += custom_target(
    _name,
    input : s,
    output : _name,
    command : [
      prog_glslang, '-V', '-x', '-o', '@OUTPUT@', '@INPUT@', glslang_quiet,
      glslang_depfile,
    ],
    depfile : f'@_name@.d',
  )
endforeach
