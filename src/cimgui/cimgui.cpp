/*
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
/* A simple C wrapper for ImGUi. */
#include <stdlib.h>

#include "cimgui.h"

#include "imgui/imgui.h"

static_assert(sizeof(ImDrawVert) == sizeof(cimgui_draw_vert));
static_assert(sizeof(ImDrawIdx) == sizeof(cimgui_draw_idx));

thread_local ImGuiContext *__MesaImGui;

cimgui_context *
imgui_create_context(uint32_t width, uint32_t height)
{
   cimgui_context *ctx;

   ctx = (cimgui_context *)malloc(sizeof(*ctx));
   if (!ctx)
      return NULL;

   ctx->context = ImGui::CreateContext();
   ImGui::SetCurrentContext(ctx->context);

   ImGui::GetIO().IniFilename = NULL;
   ImGui::GetIO().DisplaySize = ImVec2((float)width, (float)height);

   return ctx;
}

void
imgui_destroy_context(cimgui_context *ctx)
{
   ImGui::DestroyContext(ctx->context);
   free(ctx);
}

void
cimgui_get_text_data_as_rgba32(unsigned char **pixels, int *width, int *height)
{
   ImGuiIO &io = ImGui::GetIO();

   io.Fonts->GetTexDataAsRGBA32(pixels, width, height);
}

void
cimgui_set_font_texture(intptr_t font_texture_ptr)
{
   ImGuiIO &io = ImGui::GetIO();
   io.Fonts->TexID = (ImTextureID)font_texture_ptr;
}

cimgui_draw_data *
cimgui_get_draw_data()
{
   ImDrawData *imgui_data = ImGui::GetDrawData();
   cimgui_draw_data *data;

   /* Nothing to draw. */
   if (imgui_data->TotalVtxCount == 0)
      return NULL;

   data = (cimgui_draw_data *)calloc(1, +sizeof(*data));
   if (!data)
      return NULL;

   data->total_vtx_count = imgui_data->TotalVtxCount;
   data->total_idx_count = imgui_data->TotalIdxCount;
   data->display_size.x = imgui_data->DisplaySize.x;
   data->display_size.y = imgui_data->DisplaySize.y;
   data->display_pos.x = imgui_data->DisplayPos.x;
   data->display_pos.y = imgui_data->DisplayPos.y;

   data->cmd_lists_count = imgui_data->CmdListsCount;
   data->cmd_lists = (cimgui_draw_list *)calloc(imgui_data->CmdListsCount, sizeof(*data->cmd_lists));
   if (!data->cmd_lists)
      goto fail;

   for (int i = 0; i < imgui_data->CmdListsCount; i++) {
      const ImDrawList *imgui_draw_list = imgui_data->CmdLists[i];
      cimgui_draw_list *draw_list = &data->cmd_lists[i];

      draw_list->vtx_buffer.data = (cimgui_draw_vert *)imgui_draw_list->VtxBuffer.Data;
      draw_list->vtx_buffer.size = imgui_draw_list->VtxBuffer.Size;
      draw_list->idx_buffer.data = (cimgui_draw_idx *)imgui_draw_list->IdxBuffer.Data;
      draw_list->idx_buffer.size = imgui_draw_list->IdxBuffer.Size;

      draw_list->cmd_buffer.size = imgui_draw_list->CmdBuffer.Size;
      draw_list->cmd_buffer.data =
         (cimgui_draw_cmd *)calloc(imgui_draw_list->CmdBuffer.Size, sizeof(*draw_list->cmd_buffer.data));
      if (!draw_list->cmd_buffer.data)
         goto fail;

      for (int j = 0; j < imgui_draw_list->CmdBuffer.Size; j++) {
         const ImDrawCmd *imgui_cmd = &imgui_draw_list->CmdBuffer[j];
         cimgui_draw_cmd *cmd = &draw_list->cmd_buffer.data[j];

         cmd->elem_count = imgui_cmd->ElemCount;
         cmd->clip_rect.x = imgui_cmd->ClipRect.x;
         cmd->clip_rect.y = imgui_cmd->ClipRect.y;
         cmd->clip_rect.z = imgui_cmd->ClipRect.z;
         cmd->clip_rect.w = imgui_cmd->ClipRect.w;
      }
   }

   return data;
fail:
   cimgui_free_draw_data(data);
   return NULL;
}

void
cimgui_free_draw_data(cimgui_draw_data *data)
{
   for (int i = 0; i < data->cmd_lists_count; i++) {
      cimgui_draw_list *draw_list = &data->cmd_lists[i];

      free(draw_list->cmd_buffer.data);
   }
   free(data->cmd_lists);
   free(data);
}

void
cimgui_begin_rendering(cimgui_context *ctx, const cimgui_vec2 *window_size, const cimgui_vec2 *window_pos)
{
   ImGui::SetCurrentContext(ctx->context);
   ImGui::NewFrame();

   ImGui::SetNextWindowBgAlpha(0.5);
   ImGui::SetNextWindowSize(ImVec2(window_size->x, window_size->y), ImGuiCond_Always);
   ImGui::SetNextWindowPos(ImVec2(window_pos->x, window_pos->y), ImGuiCond_Always);
}

void
cimgui_end_rendering()
{
   ImGui::EndFrame();
   ImGui::Render();
}

void
cimgui_begin_panel(const char *name)
{
   ImGui::Begin(name);
}

void
cimgui_end_panel()
{
   ImGui::End();
}

void
cimgui_draw_text(const char *fmt, ...)
{
   va_list args;
   va_start(args, fmt);
   ImGui::TextV(fmt, args);
   va_end(args);
}

float
cimgui_get_cursor_pos_y()
{
   return ImGui::GetCursorPosY();
}

void
cimgui_same_line()
{
   ImGui::SameLine();
}

void
cimgui_draw_separator()
{
   ImGui::Separator();
}
