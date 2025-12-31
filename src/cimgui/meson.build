cimgui_files = files(
  'cimgui.cpp',
)

libcimgui = static_library(
  'cimgui',
  cimgui_files,
  include_directories : [
    inc_src
  ],
  dependencies : [
    libimgui_core_dep,
  ],
  gnu_symbol_visibility : 'hidden',
)
