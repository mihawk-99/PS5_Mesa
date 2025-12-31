/*
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef CIMGUI_H
#define CIMGUI_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ImGuiContext ImGuiContext;

typedef struct {
   ImGuiContext *context;
} cimgui_context;

typedef struct {
   float x;
   float y;
} cimgui_vec2;

typedef unsigned short cimgui_draw_idx;

typedef struct {
   cimgui_vec2 pos;
   cimgui_vec2 uv;
   uint32_t col;
} cimgui_draw_vert;

typedef struct {
   uint32_t elem_count;

   struct {
      float x;
      float y;
      float z;
      float w;
   } clip_rect;
} cimgui_draw_cmd;

typedef struct {
   struct {
      cimgui_draw_vert *data;
      int size;
   } vtx_buffer;

   struct {
      cimgui_draw_idx *data;
      int size;
   } idx_buffer;

   struct {
      cimgui_draw_cmd *data;
      int size;
   } cmd_buffer;
} cimgui_draw_list;

typedef struct {
   int total_vtx_count;
   int total_idx_count;

   cimgui_vec2 display_size;
   cimgui_vec2 display_pos;

   int cmd_lists_count;
   cimgui_draw_list *cmd_lists;
} cimgui_draw_data;

cimgui_context *imgui_create_context(uint32_t width, uint32_t height);
void imgui_destroy_context(cimgui_context *ctx);

void cimgui_get_text_data_as_rgba32(unsigned char **pixels, int *width, int *height);

void cimgui_set_font_texture(intptr_t font_texture_ptr);

cimgui_draw_data *cimgui_get_draw_data();
void cimgui_free_draw_data(cimgui_draw_data *data);

void cimgui_begin_rendering(cimgui_context *ctx, const cimgui_vec2 *window_size, const cimgui_vec2 *window_pos);
void cimgui_end_rendering();

void cimgui_begin_panel(const char *name);
void cimgui_end_panel();

void cimgui_draw_text(const char *fmt, ...);

float cimgui_get_cursor_pos_y();

void cimgui_same_line();

void cimgui_draw_separator();

#ifdef __cplusplus
}
#endif

#endif
