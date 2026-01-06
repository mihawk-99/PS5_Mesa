
#include "pipe/p_context.h"
#include "util/u_inlines.h"
#include "util/format/u_format.h"

#include "nouveau_context.h"
#include "nouveau_screen.h"

#include "nv50/nv50_resource.h"

static struct pipe_resource *
nv50_resource_create(struct pipe_screen *screen,
                     const struct pipe_resource *templ)
{
   switch (templ->target) {
   case PIPE_BUFFER:
      return nouveau_buffer_create(screen, templ);
   default:
      return nv50_miptree_create(screen, templ);
   }
}

static void
nv50_resource_destroy(struct pipe_screen *pscreen, struct pipe_resource *res)
{
   if (res->target == PIPE_BUFFER)
      nouveau_buffer_destroy(pscreen, res);
   else
      nv50_miptree_destroy(pscreen, res);
}

static struct pipe_resource *
nv50_resource_from_handle(struct pipe_screen * screen,
                          const struct pipe_resource *templ,
                          struct winsys_handle *whandle,
                          unsigned usage)
{
   if (templ->target == PIPE_BUFFER)
      return NULL;
   else
      return nv50_miptree_from_handle(screen, templ, whandle);
}

void
nv50_surface_destroy(struct pipe_context *pipe, struct pipe_surface *ps)
{
   struct nv50_surface *s = nv50_surface(ps);

   pipe_resource_reference(&ps->texture, NULL);

   FREE(s);
}

void
nv50_framebuffer_init(struct pipe_context *pctx,
                      const struct pipe_framebuffer_state *fb,
                      struct pipe_surface **cbufs,
                      struct pipe_surface **zsbuf)
{
   return nv_framebuffer_init(pctx, fb, cbufs, zsbuf,
                              nv50_miptree_surface_new,
                              nv50_surface_destroy);
}

void
nv50_invalidate_resource(struct pipe_context *pipe, struct pipe_resource *res)
{
   if (res->target == PIPE_BUFFER)
      nouveau_buffer_invalidate(pipe, res);
}

void
nv50_init_resource_functions(struct pipe_context *pcontext)
{
   pcontext->buffer_map = nouveau_buffer_transfer_map;
   pcontext->texture_map = nv50_miptree_transfer_map;
   pcontext->transfer_flush_region = nouveau_buffer_transfer_flush_region;
   pcontext->buffer_unmap = nouveau_buffer_transfer_unmap;
   pcontext->texture_unmap = nv50_miptree_transfer_unmap;
   pcontext->buffer_subdata = u_default_buffer_subdata;
   pcontext->texture_subdata = u_default_texture_subdata;
   pcontext->surface_destroy = nv50_surface_destroy;
   pcontext->invalidate_resource = nv50_invalidate_resource;
}

void
nv50_screen_init_resource_functions(struct pipe_screen *pscreen)
{
   pscreen->resource_create = nv50_resource_create;
   pscreen->resource_from_handle = nv50_resource_from_handle;
   pscreen->resource_get_handle = nv50_miptree_get_handle;
   pscreen->resource_destroy = nv50_resource_destroy;
}
