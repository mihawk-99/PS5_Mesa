/*
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_physical_device.h"

#include "vk_enum_to_str.h"
#include "vk_extensions.h"

#include "cimgui/cimgui.h"

#define NVK_QUEUE_GRAPHICS 0

/* Mapped from VkSwapchainKHR */
struct swapchain_data {
   VkSwapchainKHR swapchain;
   unsigned width, height;
   VkFormat format;

   uint32_t n_images;
   VkImage *images;
   VkImageView *image_views;

   VkFramebuffer *framebuffers;

   VkRenderPass render_pass;

   VkDescriptorPool descriptor_pool;
   VkDescriptorSetLayout descriptor_layout;
   VkDescriptorSet descriptor_set;

   VkSampler font_sampler;

   VkPipelineLayout pipeline_layout;
   VkPipeline pipeline;

   VkCommandPool command_pool;

   struct list_head draws; /* List of struct hud_draw */

   bool font_uploaded;
   VkImage font_image;
   VkImageView font_image_view;
   VkDeviceMemory font_mem;
   VkBuffer upload_font_buffer;
   VkDeviceMemory upload_font_buffer_mem;

   cimgui_context *ctx;
   cimgui_vec2 window_size;
};

struct hud_draw {
   struct list_head link;

   VkCommandBuffer command_buffer;

   VkSemaphore cross_engine_semaphore;

   VkSemaphore semaphore;
   VkFence fence;

   VkBuffer vertex_buffer;
   VkDeviceMemory vertex_buffer_mem;
   VkDeviceSize vertex_buffer_size;

   VkBuffer index_buffer;
   VkDeviceMemory index_buffer_mem;
   VkDeviceSize index_buffer_size;
};

static struct hash_table_u64 *vk_object_to_data = NULL;
static simple_mtx_t vk_object_to_data_mutex = SIMPLE_MTX_INITIALIZER;

static inline void
ensure_vk_object_map(void)
{
   if (!vk_object_to_data)
      vk_object_to_data = _mesa_hash_table_u64_create(NULL);
}

#define HKEY(obj)       ((uint64_t)(obj))
#define FIND(type, obj) ((type *)find_object_data(HKEY(obj)))

static void *
find_object_data(uint64_t obj)
{
   simple_mtx_lock(&vk_object_to_data_mutex);
   ensure_vk_object_map();
   void *data = _mesa_hash_table_u64_search(vk_object_to_data, obj);
   simple_mtx_unlock(&vk_object_to_data_mutex);
   return data;
}

static void
map_object(uint64_t obj, void *data)
{
   simple_mtx_lock(&vk_object_to_data_mutex);
   ensure_vk_object_map();
   _mesa_hash_table_u64_insert(vk_object_to_data, obj, data);
   simple_mtx_unlock(&vk_object_to_data_mutex);
}

static void
unmap_object(uint64_t obj)
{
   simple_mtx_lock(&vk_object_to_data_mutex);
   _mesa_hash_table_u64_remove(vk_object_to_data, obj);
   simple_mtx_unlock(&vk_object_to_data_mutex);
}

#define VK_CHECK(expr)                                                                                                 \
   do {                                                                                                                \
      VkResult __result = (expr);                                                                                      \
      if (__result != VK_SUCCESS) {                                                                                    \
         fprintf(stderr, "'%s' line %i failed with %s\n", #expr, __LINE__, vk_Result_to_str(__result));                \
      }                                                                                                                \
   } while (0)

/* Swapchain data */
static struct swapchain_data *
new_swapchain_data(VkSwapchainKHR swapchain)
{
   struct swapchain_data *data = rzalloc(NULL, struct swapchain_data);
   data->swapchain = swapchain;
   data->window_size.x = 320;
   data->window_size.y = 600;
   list_inithead(&data->draws);
   map_object(HKEY(data->swapchain), data);

   return data;
}

static void
shutdown_swapchain_data(struct nvk_device *device, struct swapchain_data *data)
{
   list_for_each_entry_safe (struct hud_draw, draw, &data->draws, link) {
      device->layer_dispatch.hud.DestroySemaphore(nvk_device_to_handle(device), draw->cross_engine_semaphore, NULL);
      device->layer_dispatch.hud.DestroySemaphore(nvk_device_to_handle(device), draw->semaphore, NULL);
      device->layer_dispatch.hud.DestroyFence(nvk_device_to_handle(device), draw->fence, NULL);
      device->layer_dispatch.hud.DestroyBuffer(nvk_device_to_handle(device), draw->vertex_buffer, NULL);
      device->layer_dispatch.hud.DestroyBuffer(nvk_device_to_handle(device), draw->index_buffer, NULL);
      device->layer_dispatch.hud.FreeMemory(nvk_device_to_handle(device), draw->vertex_buffer_mem, NULL);
      device->layer_dispatch.hud.FreeMemory(nvk_device_to_handle(device), draw->index_buffer_mem, NULL);
   }

   for (uint32_t i = 0; i < data->n_images; i++) {
      device->layer_dispatch.hud.DestroyImageView(nvk_device_to_handle(device), data->image_views[i], NULL);
      device->layer_dispatch.hud.DestroyFramebuffer(nvk_device_to_handle(device), data->framebuffers[i], NULL);
   }

   device->layer_dispatch.hud.DestroyRenderPass(nvk_device_to_handle(device), data->render_pass, NULL);

   device->layer_dispatch.hud.DestroyCommandPool(nvk_device_to_handle(device), data->command_pool, NULL);

   device->layer_dispatch.hud.DestroyPipeline(nvk_device_to_handle(device), data->pipeline, NULL);
   device->layer_dispatch.hud.DestroyPipelineLayout(nvk_device_to_handle(device), data->pipeline_layout, NULL);

   device->layer_dispatch.hud.DestroyDescriptorPool(nvk_device_to_handle(device), data->descriptor_pool, NULL);
   device->layer_dispatch.hud.DestroyDescriptorSetLayout(nvk_device_to_handle(device), data->descriptor_layout, NULL);

   device->layer_dispatch.hud.DestroySampler(nvk_device_to_handle(device), data->font_sampler, NULL);
   device->layer_dispatch.hud.DestroyImageView(nvk_device_to_handle(device), data->font_image_view, NULL);
   device->layer_dispatch.hud.DestroyImage(nvk_device_to_handle(device), data->font_image, NULL);
   device->layer_dispatch.hud.FreeMemory(nvk_device_to_handle(device), data->font_mem, NULL);

   device->layer_dispatch.hud.DestroyBuffer(nvk_device_to_handle(device), data->upload_font_buffer, NULL);
   device->layer_dispatch.hud.FreeMemory(nvk_device_to_handle(device), data->upload_font_buffer_mem, NULL);

   imgui_destroy_context(data->ctx);
}

static void
destroy_swapchain_data(struct swapchain_data *data)
{
   unmap_object(HKEY(data->swapchain));
   ralloc_free(data);
}

static uint32_t
vk_memory_type(struct nvk_device *device, VkMemoryPropertyFlags properties, uint32_t type_bits)
{
   struct nvk_physical_device *pdev = (struct nvk_physical_device*)nvk_device_physical(device);
   VkPhysicalDeviceMemoryProperties prop;

   pdev->vk.dispatch_table.GetPhysicalDeviceMemoryProperties(nvk_physical_device_to_handle(pdev), &prop);

   for (uint32_t i = 0; i < prop.memoryTypeCount; i++)
      if ((prop.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
         return i;
   return 0xFFFFFFFF; /* Unable to find memoryType */
}

static const uint32_t hud_vert_spv[] = {
#include "hud.vert.spv.h"
};
static const uint32_t hud_frag_spv[] = {
#include "hud.frag.spv.h"
};

static void
setup_swapchain_data_pipeline(struct nvk_device *device, struct swapchain_data *data)
{
   VkShaderModule vert_module, frag_module;

   /* Create shader modules */
   VkShaderModuleCreateInfo smci_vert = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(hud_vert_spv),
      .pCode = hud_vert_spv,
   };
   VK_CHECK(
      device->layer_dispatch.hud.CreateShaderModule(nvk_device_to_handle(device), &smci_vert, NULL, &vert_module));

   VkShaderModuleCreateInfo smci_frag = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(hud_frag_spv),
      .pCode = (uint32_t *)hud_frag_spv,
   };
   VK_CHECK(
      device->layer_dispatch.hud.CreateShaderModule(nvk_device_to_handle(device), &smci_frag, NULL, &frag_module));

   /* Font sampler */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .maxAnisotropy = 1.0f,
      .minLod = -1000,
      .maxLod = 1000,
   };
   VK_CHECK(device->layer_dispatch.hud.CreateSampler(nvk_device_to_handle(device), &sci, NULL, &data->font_sampler));

   /* Descriptor pool */
   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
   };

   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &dps,
   };
   VK_CHECK(device->layer_dispatch.hud.CreateDescriptorPool(nvk_device_to_handle(device), &dpci, NULL,
                                                            &data->descriptor_pool));

   /* Descriptor layout */
   VkSampler sampler[1] = {data->font_sampler};
   VkDescriptorSetLayoutBinding binding[1] = {};
   binding[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   binding[0].descriptorCount = 1;
   binding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   binding[0].pImmutableSamplers = sampler;

   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = binding,
   };
   VK_CHECK(device->layer_dispatch.hud.CreateDescriptorSetLayout(nvk_device_to_handle(device), &dslci, NULL,
                                                                 &data->descriptor_layout));

   /* Descriptor set */
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = data->descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &data->descriptor_layout,
   };
   VK_CHECK(
      device->layer_dispatch.hud.AllocateDescriptorSets(nvk_device_to_handle(device), &dsai, &data->descriptor_set));

   /* Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection matrix
    */
   VkPushConstantRange pc = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = sizeof(float) * 0,
      .size = sizeof(float) * 4,
   };

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &data->descriptor_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pc,
   };
   VK_CHECK(device->layer_dispatch.hud.CreatePipelineLayout(nvk_device_to_handle(device), &plci, NULL,
                                                            &data->pipeline_layout));

   VkPipelineShaderStageCreateInfo stage[2] = {};
   stage[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   stage[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
   stage[0].module = vert_module;
   stage[0].pName = "main";
   stage[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   stage[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   stage[1].module = frag_module;
   stage[1].pName = "main";

   VkVertexInputBindingDescription binding_desc[1] = {};
   binding_desc[0].stride = sizeof(cimgui_draw_vert);
   binding_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

   VkVertexInputAttributeDescription attribute_desc[3] = {};
   attribute_desc[0].location = 0;
   attribute_desc[0].binding = binding_desc[0].binding;
   attribute_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
   attribute_desc[0].offset = offsetof(cimgui_draw_vert, pos);
   attribute_desc[1].location = 1;
   attribute_desc[1].binding = binding_desc[0].binding;
   attribute_desc[1].format = VK_FORMAT_R32G32_SFLOAT;
   attribute_desc[1].offset = offsetof(cimgui_draw_vert, uv);
   attribute_desc[2].location = 2;
   attribute_desc[2].binding = binding_desc[0].binding;
   attribute_desc[2].format = VK_FORMAT_R8G8B8A8_UNORM;
   attribute_desc[2].offset = offsetof(cimgui_draw_vert, col);

   VkPipelineVertexInputStateCreateInfo vertex_info = {};
   vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
   vertex_info.vertexBindingDescriptionCount = 1;
   vertex_info.pVertexBindingDescriptions = binding_desc;
   vertex_info.vertexAttributeDescriptionCount = 3;
   vertex_info.pVertexAttributeDescriptions = attribute_desc;

   VkPipelineInputAssemblyStateCreateInfo ia_info = {};
   ia_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
   ia_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   VkPipelineViewportStateCreateInfo viewport_info = {};
   viewport_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
   viewport_info.viewportCount = 1;
   viewport_info.scissorCount = 1;

   VkPipelineRasterizationStateCreateInfo raster_info = {};
   raster_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
   raster_info.polygonMode = VK_POLYGON_MODE_FILL;
   raster_info.cullMode = VK_CULL_MODE_NONE;
   raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   raster_info.lineWidth = 1.0f;

   VkPipelineMultisampleStateCreateInfo ms_info = {};
   ms_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
   ms_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

   VkPipelineColorBlendAttachmentState color_attachment[1] = {};
   color_attachment[0].blendEnable = VK_TRUE;
   color_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
   color_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
   color_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
   color_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
   color_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
   color_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;
   color_attachment[0].colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

   VkPipelineDepthStencilStateCreateInfo depth_info = {};
   depth_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

   VkPipelineColorBlendStateCreateInfo blend_info = {};
   blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
   blend_info.attachmentCount = 1;
   blend_info.pAttachments = color_attachment;

   VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dynamic_state = {};
   dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
   dynamic_state.dynamicStateCount = ARRAY_SIZE(dynamic_states);
   dynamic_state.pDynamicStates = dynamic_states;

   VkGraphicsPipelineCreateInfo info = {};
   info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
   info.flags = 0;
   info.stageCount = 2;
   info.pStages = stage;
   info.pVertexInputState = &vertex_info;
   info.pInputAssemblyState = &ia_info;
   info.pViewportState = &viewport_info;
   info.pRasterizationState = &raster_info;
   info.pMultisampleState = &ms_info;
   info.pDepthStencilState = &depth_info;
   info.pColorBlendState = &blend_info;
   info.pDynamicState = &dynamic_state;
   info.layout = data->pipeline_layout;
   info.renderPass = data->render_pass;
   VK_CHECK(device->layer_dispatch.hud.CreateGraphicsPipelines(nvk_device_to_handle(device), VK_NULL_HANDLE, 1, &info,
                                                               NULL, &data->pipeline));

   device->layer_dispatch.hud.DestroyShaderModule(nvk_device_to_handle(device), vert_module, NULL);
   device->layer_dispatch.hud.DestroyShaderModule(nvk_device_to_handle(device), frag_module, NULL);

   unsigned char *pixels;
   int width, height;

   cimgui_get_text_data_as_rgba32(&pixels, &width, &height);

   /* Font image */
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent =
         (VkExtent3D){
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .depth = 1,
         },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VK_CHECK(device->layer_dispatch.hud.CreateImage(nvk_device_to_handle(device), &ici, NULL, &data->font_image));

   VkImageMemoryRequirementsInfo2 image_req_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = data->font_image,
   };
   VkMemoryRequirements2 font_image_req = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   device->layer_dispatch.hud.GetImageMemoryRequirements2(nvk_device_to_handle(device), &image_req_info,
                                                          &font_image_req);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = font_image_req.memoryRequirements.size,
      .memoryTypeIndex =
         vk_memory_type(device, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, font_image_req.memoryRequirements.memoryTypeBits),
   };
   VK_CHECK(device->layer_dispatch.hud.AllocateMemory(nvk_device_to_handle(device), &mai, NULL, &data->font_mem));

   VkBindImageMemoryInfo bimi = {
      .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
      .image = data->font_image,
      .memory = data->font_mem,
   };
   VK_CHECK(device->layer_dispatch.hud.BindImageMemory2(nvk_device_to_handle(device), 1, &bimi));

   /* Font image view */
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = data->font_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange =
         (VkImageSubresourceRange){
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
         },
   };
   VK_CHECK(
      device->layer_dispatch.hud.CreateImageView(nvk_device_to_handle(device), &ivci, NULL, &data->font_image_view));

   /* Descriptor set */
   VkDescriptorImageInfo desc_image[1] = {};
   desc_image[0].sampler = data->font_sampler;
   desc_image[0].imageView = data->font_image_view;
   desc_image[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = data->descriptor_set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = desc_image,
   };
   device->layer_dispatch.hud.UpdateDescriptorSets(nvk_device_to_handle(device), 1, &wds, 0, NULL);
}

static void
setup_swapchain_data(struct nvk_device *device, struct swapchain_data *data,
                     const VkSwapchainCreateInfoKHR *pCreateInfo)
{
   data->width = pCreateInfo->imageExtent.width;
   data->height = pCreateInfo->imageExtent.height;
   data->format = pCreateInfo->imageFormat;

   data->ctx = imgui_create_context(data->width, data->height);

   /* Render pass */
   VkAttachmentDescription attachment_desc = {};
   attachment_desc.format = pCreateInfo->imageFormat;
   attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
   attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   attachment_desc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   attachment_desc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
   VkAttachmentReference color_attachment = {};
   color_attachment.attachment = 0;
   color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   VkSubpassDescription subpass = {};
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   subpass.colorAttachmentCount = 1;
   subpass.pColorAttachments = &color_attachment;
   VkSubpassDependency dependency = {};
   dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
   dependency.dstSubpass = 0;
   dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
   dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
   dependency.srcAccessMask = 0;
   dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment_desc,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency,
   };

   VK_CHECK(
      device->layer_dispatch.hud.CreateRenderPass(nvk_device_to_handle(device), &rpci, NULL, &data->render_pass));

   setup_swapchain_data_pipeline(device, data);

   data->images = ralloc_array(data, VkImage, data->n_images);
   data->image_views = ralloc_array(data, VkImageView, data->n_images);
   data->framebuffers = ralloc_array(data, VkFramebuffer, data->n_images);

   VK_CHECK(device->layer_dispatch.hud.GetSwapchainImagesKHR(nvk_device_to_handle(device), data->swapchain,
                                                             &data->n_images, NULL));

   data->images = ralloc_array(data, VkImage, data->n_images);
   data->image_views = ralloc_array(data, VkImageView, data->n_images);
   data->framebuffers = ralloc_array(data, VkFramebuffer, data->n_images);

   VK_CHECK(device->layer_dispatch.hud.GetSwapchainImagesKHR(nvk_device_to_handle(device), data->swapchain,
                                                             &data->n_images, data->images));

   /* Image views */
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = pCreateInfo->imageFormat,
      .components =
         (VkComponentMapping){
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
         },
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };

   for (uint32_t i = 0; i < data->n_images; i++) {
      ivci.image = data->images[i];
      VK_CHECK(
         device->layer_dispatch.hud.CreateImageView(nvk_device_to_handle(device), &ivci, NULL, &data->image_views[i]));
   }

   /* Framebuffers */
   VkImageView attachment[1];
   VkFramebufferCreateInfo fci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = data->render_pass,
      .attachmentCount = 1,
      .pAttachments = attachment,
      .width = data->width,
      .height = data->height,
      .layers = 1,
   };

   for (uint32_t i = 0; i < data->n_images; i++) {
      attachment[0] = data->image_views[i];
      VK_CHECK(device->layer_dispatch.hud.CreateFramebuffer(nvk_device_to_handle(device), &fci, NULL,
                                                            &data->framebuffers[i]));
   }

   /* Command buffer pool */
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = NVK_QUEUE_GRAPHICS,
   };

   VK_CHECK(
      device->layer_dispatch.hud.CreateCommandPool(nvk_device_to_handle(device), &cpci, NULL, &data->command_pool));
}

static void
compute_swapchain_display(struct nvk_device *device, struct swapchain_data *data)
{
   struct nvk_physical_device *pdev = (struct nvk_physical_device*)nvk_device_physical(device);
   const float margin = 10.0f;

   VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
   };
   VkPhysicalDeviceMemoryProperties2 mem_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
      .pNext = &budget_props,
   };
   pdev->vk.dispatch_table.GetPhysicalDeviceMemoryProperties2(nvk_physical_device_to_handle(pdev), &mem_props);

   cimgui_vec2 window_pos;
   window_pos.x = data->width - data->window_size.x - margin;
   window_pos.y = margin;

   cimgui_begin_rendering(data->ctx, &data->window_size, &window_pos);
   cimgui_begin_panel("NVK HUD");
   cimgui_draw_text("Driver: %s", pdev->vk.properties.driverInfo);
   cimgui_draw_text("Device: %s", pdev->vk.properties.deviceName);
   cimgui_draw_separator();

   cimgui_draw_text("Memory info (in MiB):");
   for (uint32_t i = 0; i < mem_props.memoryProperties.memoryHeapCount; i++) {
      VkMemoryHeap heap = mem_props.memoryProperties.memoryHeaps[i];
      const uint64_t budget = budget_props.heapBudget[i] / (1024 * 1024);
      const uint64_t usage = budget_props.heapUsage[i] / (1024 * 1024);

      if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
         cimgui_draw_text("VRAM: %" PRId64 " (budget), %" PRId64 " (usage)", budget, usage);
      } else {
         cimgui_draw_text("GTT: %" PRId64 " (budget), %" PRId64 " (usage)", budget, usage);
      }
   }

   cimgui_draw_separator();
   cimgui_draw_text("Enabled extensions:");
   uint32_t idx;
   for (idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
      if (device->vk.enabled_extensions.extensions[idx])
         cimgui_draw_text(" - %s", vk_device_extensions[idx].extensionName);
   }   

   data->window_size.y = cimgui_get_cursor_pos_y() + margin;

   cimgui_end_panel();
   cimgui_end_rendering();
}

static struct hud_draw *
get_hud_draw(struct nvk_device *device, struct swapchain_data *data)
{
   struct hud_draw *draw = list_is_empty(&data->draws) ? NULL : list_first_entry(&data->draws, struct hud_draw, link);

   VkSemaphoreCreateInfo sci = {};
   sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

   if (draw && device->layer_dispatch.hud.GetFenceStatus(nvk_device_to_handle(device), draw->fence) == VK_SUCCESS) {
      list_del(&draw->link);
      VK_CHECK(device->layer_dispatch.hud.ResetFences(nvk_device_to_handle(device), 1, &draw->fence));
      list_addtail(&draw->link, &data->draws);
      return draw;
   }

   draw = rzalloc(data, struct hud_draw);

   VkCommandBufferAllocateInfo cbai = {};
   cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cbai.commandPool = data->command_pool;
   cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cbai.commandBufferCount = 1;
   VK_CHECK(
      device->layer_dispatch.hud.AllocateCommandBuffers(nvk_device_to_handle(device), &cbai, &draw->command_buffer));

   VkFenceCreateInfo fence_info = {};
   fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
   VK_CHECK(device->layer_dispatch.hud.CreateFence(nvk_device_to_handle(device), &fence_info, NULL, &draw->fence));

   VK_CHECK(device->layer_dispatch.hud.CreateSemaphore(nvk_device_to_handle(device), &sci, NULL, &draw->semaphore));
   VK_CHECK(device->layer_dispatch.hud.CreateSemaphore(nvk_device_to_handle(device), &sci, NULL,
                                                       &draw->cross_engine_semaphore));

   list_addtail(&draw->link, &data->draws);

   return draw;
}

static void
ensure_swapchain_fonts(struct nvk_device *device, struct swapchain_data *data, VkCommandBuffer command_buffer)
{
   if (data->font_uploaded)
      return;

   data->font_uploaded = true;

   unsigned char *pixels;
   int width, height;

   cimgui_get_text_data_as_rgba32(&pixels, &width, &height);

   size_t upload_size = width * height * 4 * sizeof(char);

   /* Upload buffer */
   VkBufferUsageFlags2CreateInfoKHR usage2_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
      .pNext = NULL,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
   };
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &usage2_info,
      .size = upload_size,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_CHECK(
      device->layer_dispatch.hud.CreateBuffer(nvk_device_to_handle(device), &bci, NULL, &data->upload_font_buffer));

   VkDeviceBufferMemoryRequirements dbmr = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pNext = NULL,
      .pCreateInfo = &bci,
   };
   VkMemoryRequirements2 upload_buffer_req = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   device->layer_dispatch.hud.GetDeviceBufferMemoryRequirements(nvk_device_to_handle(device), &dbmr,
                                                                &upload_buffer_req);

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = upload_buffer_req.memoryRequirements.size,
      .memoryTypeIndex = vk_memory_type(device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        upload_buffer_req.memoryRequirements.memoryTypeBits),
   };
   VK_CHECK(device->layer_dispatch.hud.AllocateMemory(nvk_device_to_handle(device), &mai, NULL,
                                                      &data->upload_font_buffer_mem));

   VkBindBufferMemoryInfo bbmi = {
      .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
      .buffer = data->upload_font_buffer,
      .memory = data->upload_font_buffer_mem,
   };

   VK_CHECK(device->layer_dispatch.hud.BindBufferMemory2(nvk_device_to_handle(device), 1, &bbmi));

   /* Upload to Buffer */
   char *map = NULL;
   VK_CHECK(device->layer_dispatch.hud.MapMemory(nvk_device_to_handle(device), data->upload_font_buffer_mem, 0,
                                                 upload_size, 0, (void **)(&map)));
   memcpy(map, pixels, upload_size);
   VkMappedMemoryRange range[1] = {};
   range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
   range[0].memory = data->upload_font_buffer_mem;
   range[0].size = upload_size;
   VK_CHECK(device->layer_dispatch.hud.FlushMappedMemoryRanges(nvk_device_to_handle(device), 1, range));
   device->layer_dispatch.hud.UnmapMemory(nvk_device_to_handle(device), data->upload_font_buffer_mem);

   /* Copy buffer to image */
   VkImageMemoryBarrier copy_barrier[1] = {};
   copy_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
   copy_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   copy_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   copy_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   copy_barrier[0].image = data->font_image;
   copy_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   copy_barrier[0].subresourceRange.levelCount = 1;
   copy_barrier[0].subresourceRange.layerCount = 1;
   device->layer_dispatch.hud.CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, copy_barrier);

   VkBufferImageCopy region = {};
   region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   region.imageSubresource.layerCount = 1;
   region.imageExtent.width = width;
   region.imageExtent.height = height;
   region.imageExtent.depth = 1;
   device->layer_dispatch.hud.CmdCopyBufferToImage(command_buffer, data->upload_font_buffer, data->font_image,
                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

   VkImageMemoryBarrier use_barrier[1] = {};
   use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
   use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   use_barrier[0].image = data->font_image;
   use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   use_barrier[0].subresourceRange.levelCount = 1;
   use_barrier[0].subresourceRange.layerCount = 1;
   device->layer_dispatch.hud.CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                                                 use_barrier);

   /* Store our identifier */
   cimgui_set_font_texture((intptr_t)data->font_image);
}

static void
create_or_resize_buffer(struct nvk_device *device, VkBuffer *buffer, VkDeviceMemory *buffer_memory,
                        VkDeviceSize *buffer_size, size_t new_size, VkBufferUsageFlagBits usage)
{
   if (*buffer != VK_NULL_HANDLE)
      device->layer_dispatch.hud.DestroyBuffer(nvk_device_to_handle(device), *buffer, NULL);
   if (*buffer_memory)
      device->layer_dispatch.hud.FreeMemory(nvk_device_to_handle(device), *buffer_memory, NULL);

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = new_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_CHECK(device->layer_dispatch.hud.CreateBuffer(nvk_device_to_handle(device), &bci, NULL, buffer));

   VkMemoryRequirements req;
   device->layer_dispatch.hud.GetBufferMemoryRequirements(nvk_device_to_handle(device), *buffer, &req);

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = vk_memory_type(device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, req.memoryTypeBits),
   };

   VK_CHECK(device->layer_dispatch.hud.AllocateMemory(nvk_device_to_handle(device), &mai, NULL, buffer_memory));

   VkBindBufferMemoryInfo bbmi = {
      .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
      .buffer = *buffer,
      .memory = *buffer_memory,
   };
   VK_CHECK(device->layer_dispatch.hud.BindBufferMemory2(nvk_device_to_handle(device), 1, &bbmi));

   *buffer_size = new_size;
}

static struct hud_draw *
render_swapchain_display(struct nvk_queue *queue, struct swapchain_data *data, const VkSemaphore *wait_semaphores,
                         unsigned n_wait_semaphores, unsigned image_index)
{
   struct nvk_device *device = nvk_queue_device(queue);
   cimgui_draw_data *draw_data;

   draw_data = cimgui_get_draw_data();
   if (!draw_data)
      return NULL;

   struct hud_draw *draw = get_hud_draw(device, data);

   device->layer_dispatch.hud.ResetCommandBuffer(draw->command_buffer, 0);

   VkRenderPassBeginInfo render_pass_info = {};
   render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   render_pass_info.renderPass = data->render_pass;
   render_pass_info.framebuffer = data->framebuffers[image_index];
   render_pass_info.renderArea.extent.width = data->width;
   render_pass_info.renderArea.extent.height = data->height;

   VkCommandBufferBeginInfo buffer_begin_info = {};
   buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

   device->layer_dispatch.hud.BeginCommandBuffer(draw->command_buffer, &buffer_begin_info);

   ensure_swapchain_fonts(device, data, draw->command_buffer);

   /* Bounce the image to display back to color attachment layout for rendering on top of it.
    */
   VkImageMemoryBarrier imb;
   imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
   imb.pNext = NULL;
   imb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   imb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   imb.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
   imb.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   imb.image = data->images[image_index];
   imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   imb.subresourceRange.baseMipLevel = 0;
   imb.subresourceRange.levelCount = 1;
   imb.subresourceRange.baseArrayLayer = 0;
   imb.subresourceRange.layerCount = 1;
   imb.srcQueueFamilyIndex = queue->vk.queue_family_index;
   imb.dstQueueFamilyIndex = NVK_QUEUE_GRAPHICS;
   device->layer_dispatch.hud.CmdPipelineBarrier(draw->command_buffer, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                                 VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, NULL, 0, NULL, 1, &imb);

   device->layer_dispatch.hud.CmdBeginRenderPass(draw->command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

   /* Create/Resize vertex & index buffers */
   size_t vertex_size = align_uintptr(draw_data->total_vtx_count * sizeof(cimgui_draw_vert), 64);
   size_t index_size = align_uintptr(draw_data->total_idx_count * sizeof(cimgui_draw_idx), 64);
   if (draw->vertex_buffer_size < vertex_size) {
      create_or_resize_buffer(device, &draw->vertex_buffer, &draw->vertex_buffer_mem, &draw->vertex_buffer_size,
                              vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
   }
   if (draw->index_buffer_size < index_size) {
      create_or_resize_buffer(device, &draw->index_buffer, &draw->index_buffer_mem, &draw->index_buffer_size,
                              index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   }

   /* Upload vertex & index data */
   cimgui_draw_vert *vtx_dst = NULL;
   cimgui_draw_idx *idx_dst = NULL;
   VK_CHECK(device->layer_dispatch.hud.MapMemory(nvk_device_to_handle(device), draw->vertex_buffer_mem, 0, vertex_size,
                                                 0, (void **)(&vtx_dst)));
   VK_CHECK(device->layer_dispatch.hud.MapMemory(nvk_device_to_handle(device), draw->index_buffer_mem, 0, index_size,
                                                 0, (void **)(&idx_dst)));

   for (int i = 0; i < draw_data->cmd_lists_count; i++) {
      const cimgui_draw_list *draw_list = &draw_data->cmd_lists[i];

      memcpy(vtx_dst, draw_list->vtx_buffer.data, draw_list->vtx_buffer.size * sizeof(cimgui_draw_vert));
      memcpy(idx_dst, draw_list->idx_buffer.data, draw_list->idx_buffer.size * sizeof(cimgui_draw_idx));

      vtx_dst += draw_list->vtx_buffer.size;
      idx_dst += draw_list->idx_buffer.size;
   }

   VkMappedMemoryRange range[2] = {};
   range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
   range[0].memory = draw->vertex_buffer_mem;
   range[0].size = VK_WHOLE_SIZE;
   range[1].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
   range[1].memory = draw->index_buffer_mem;
   range[1].size = VK_WHOLE_SIZE;
   VK_CHECK(device->layer_dispatch.hud.FlushMappedMemoryRanges(nvk_device_to_handle(device), 2, range));
   device->layer_dispatch.hud.UnmapMemory(nvk_device_to_handle(device), draw->vertex_buffer_mem);
   device->layer_dispatch.hud.UnmapMemory(nvk_device_to_handle(device), draw->index_buffer_mem);

   /* Bind pipeline and descriptor sets */
   device->layer_dispatch.hud.CmdBindPipeline(draw->command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, data->pipeline);
   VkDescriptorSet desc_set[1] = {data->descriptor_set};
   device->layer_dispatch.hud.CmdBindDescriptorSets(draw->command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                    data->pipeline_layout, 0, 1, desc_set, 0, NULL);

   /* Bind vertex & index buffers */
   VkBuffer vertex_buffers[1] = {draw->vertex_buffer};
   VkDeviceSize vertex_offset[1] = {0};
   device->layer_dispatch.hud.CmdBindVertexBuffers(draw->command_buffer, 0, 1, vertex_buffers, vertex_offset);
   device->layer_dispatch.hud.CmdBindIndexBuffer(draw->command_buffer, draw->index_buffer, 0, VK_INDEX_TYPE_UINT16);

   /* Setup viewport */
   VkViewport viewport;
   viewport.x = 0;
   viewport.y = 0;
   viewport.width = draw_data->display_size.x;
   viewport.height = draw_data->display_size.y;
   viewport.minDepth = 0.0f;
   viewport.maxDepth = 1.0f;
   device->layer_dispatch.hud.CmdSetViewport(draw->command_buffer, 0, 1, &viewport);

   /* Setup scale and translation through push constants :
    *
    * Our visible imgui space lies from draw_data->display_pos (top left) to
    * draw_data->display_pos+draw_data->display_size (bottom right). DisplayMin
    * is typically (0,0) for single viewport apps.
    */
   float scale[2];
   scale[0] = 2.0f / draw_data->display_size.x;
   scale[1] = 2.0f / draw_data->display_size.y;
   float translate[2];
   translate[0] = -1.0f - draw_data->display_pos.x * scale[0];
   translate[1] = -1.0f - draw_data->display_pos.y * scale[1];
   device->layer_dispatch.hud.CmdPushConstants(draw->command_buffer, data->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                                               sizeof(float) * 0, sizeof(float) * 2, scale);
   device->layer_dispatch.hud.CmdPushConstants(draw->command_buffer, data->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                                               sizeof(float) * 2, sizeof(float) * 2, translate);

   /* Rendering */
   int vtx_offset = 0;
   int idx_offset = 0;
   cimgui_vec2 display_pos = draw_data->display_pos;
   for (int n = 0; n < draw_data->cmd_lists_count; n++) {
      const cimgui_draw_list *cmd_list = &draw_data->cmd_lists[n];

      for (int cmd_i = 0; cmd_i < cmd_list->cmd_buffer.size; cmd_i++) {
         const cimgui_draw_cmd *cmd = &cmd_list->cmd_buffer.data[cmd_i];

         // Apply scissor/clipping rectangle
         // FIXME: We could clamp width/height based on clamped min/max values.
         VkRect2D scissor;
         scissor.offset.x =
            (int32_t)(cmd->clip_rect.x - display_pos.x) > 0 ? (int32_t)(cmd->clip_rect.x - display_pos.x) : 0;
         scissor.offset.y =
            (int32_t)(cmd->clip_rect.y - display_pos.y) > 0 ? (int32_t)(cmd->clip_rect.y - display_pos.y) : 0;
         scissor.extent.width = (uint32_t)(cmd->clip_rect.z - cmd->clip_rect.x);
         scissor.extent.height = (uint32_t)(cmd->clip_rect.w - cmd->clip_rect.y + 1); // FIXME: Why +1 here?
         device->layer_dispatch.hud.CmdSetScissor(draw->command_buffer, 0, 1, &scissor);

         device->layer_dispatch.hud.CmdDrawIndexed(draw->command_buffer, cmd->elem_count, 1, idx_offset, vtx_offset, 0);

         idx_offset += cmd->elem_count;
      }

      vtx_offset += cmd_list->vtx_buffer.size;
   }
   device->layer_dispatch.hud.CmdEndRenderPass(draw->command_buffer);

   if (queue->vk.queue_family_index != NVK_QUEUE_GRAPHICS) {
      /* Transfer the image back to the present queue family image layout was already changed to
       * present by the render pass
       */
      imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imb.pNext = NULL;
      imb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      imb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      imb.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      imb.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      imb.image = data->images[image_index];
      imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imb.subresourceRange.baseMipLevel = 0;
      imb.subresourceRange.levelCount = 1;
      imb.subresourceRange.baseArrayLayer = 0;
      imb.subresourceRange.layerCount = 1;
      imb.srcQueueFamilyIndex = NVK_QUEUE_GRAPHICS;
      imb.dstQueueFamilyIndex = queue->vk.queue_family_index;
      device->layer_dispatch.hud.CmdPipelineBarrier(draw->command_buffer, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                                    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, NULL, 0, NULL, 1, &imb);
   }

   device->layer_dispatch.hud.EndCommandBuffer(draw->command_buffer);

   /* When presenting on a different queue than where we're drawing the hud *AND* when the
    * application does not provide a semaphore to vkQueuePresent, insert our own cross engine
    * synchronization semaphore.
    */
   if (n_wait_semaphores == 0 && queue->vk.queue_family_index != NVK_QUEUE_GRAPHICS) {
      VkPipelineStageFlags stages_wait = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo submit_info = {};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 0;
      submit_info.pWaitDstStageMask = &stages_wait;
      submit_info.waitSemaphoreCount = 0;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &draw->cross_engine_semaphore;

      device->layer_dispatch.hud.QueueSubmit(nvk_queue_to_handle(queue), 1, &submit_info, VK_NULL_HANDLE);

      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 1;
      submit_info.pWaitDstStageMask = &stages_wait;
      submit_info.pCommandBuffers = &draw->command_buffer;
      submit_info.waitSemaphoreCount = 1;
      submit_info.pWaitSemaphores = &draw->cross_engine_semaphore;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &draw->semaphore;

      device->layer_dispatch.hud.QueueSubmit(nvk_queue_to_handle(device->gfx_queue), 1, &submit_info,
                                             draw->fence);
   } else {
      VkPipelineStageFlags *stages_wait =
         (VkPipelineStageFlags *)malloc(sizeof(VkPipelineStageFlags) * n_wait_semaphores);
      for (unsigned i = 0; i < n_wait_semaphores; i++) {
         /* Wait in the fragment stage until the swapchain image is ready. */
         stages_wait[i] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }

      VkSubmitInfo submit_info = {};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &draw->command_buffer;
      submit_info.pWaitDstStageMask = stages_wait;
      submit_info.waitSemaphoreCount = n_wait_semaphores;
      submit_info.pWaitSemaphores = wait_semaphores;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &draw->semaphore;

      device->layer_dispatch.hud.QueueSubmit(nvk_queue_to_handle(device->gfx_queue), 1, &submit_info,
                                             draw->fence);

      free(stages_wait);
   }

   cimgui_free_draw_data(draw_data);
   return draw;
}

VKAPI_ATTR VkResult VKAPI_CALL
hud_CreateSwapchainKHR(VkDevice _device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchain)
{
   VK_FROM_HANDLE(nvk_device, device, _device);

   VkResult result = device->layer_dispatch.hud.CreateSwapchainKHR(_device, pCreateInfo, pAllocator, pSwapchain);
   if (result != VK_SUCCESS)
      return result;

   struct swapchain_data *swapchain_data = new_swapchain_data(*pSwapchain);
   setup_swapchain_data(device, swapchain_data, pCreateInfo);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hud_DestroySwapchainKHR(VkDevice _device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(nvk_device, device, _device);

   if (swapchain == VK_NULL_HANDLE) {
      device->layer_dispatch.hud.DestroySwapchainKHR(_device, swapchain, pAllocator);
      return;
   }

   struct swapchain_data *swapchain_data = FIND(struct swapchain_data, swapchain);

   shutdown_swapchain_data(device, swapchain_data);
   device->layer_dispatch.hud.DestroySwapchainKHR(_device, swapchain, pAllocator);
   destroy_swapchain_data(swapchain_data);
}

VKAPI_ATTR VkResult VKAPI_CALL
hud_QueuePresentKHR(VkQueue _queue, const VkPresentInfoKHR *pPresentInfo)
{
   VK_FROM_HANDLE(nvk_queue, queue, _queue);
   struct nvk_device *device = nvk_queue_device(queue);
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++) {
      VkSwapchainKHR swapchain = pPresentInfo->pSwapchains[i];
      struct swapchain_data *swapchain_data = FIND(struct swapchain_data, swapchain);

      uint32_t image_index = pPresentInfo->pImageIndices[i];

      VkPresentInfoKHR present_info = *pPresentInfo;
      present_info.swapchainCount = 1;
      present_info.pSwapchains = &swapchain;
      present_info.pImageIndices = &image_index;

      compute_swapchain_display(device, swapchain_data);

      struct hud_draw *draw = render_swapchain_display(queue, swapchain_data, pPresentInfo->pWaitSemaphores,
                                                       pPresentInfo->waitSemaphoreCount, image_index);

      /* Because the submission of the hud draw waits on the semaphores handed for present,
       * we don't need to have this present operation wait on them as well, we can just wait on
       * the hud submission semaphore.
       */
      present_info.pWaitSemaphores = &draw->semaphore;
      present_info.waitSemaphoreCount = 1;

      VkResult chain_result = device->layer_dispatch.hud.QueuePresentKHR(_queue, &present_info);
      if (pPresentInfo->pResults)
         pPresentInfo->pResults[i] = chain_result;
      if (chain_result != VK_SUCCESS && result == VK_SUCCESS)
         result = chain_result;
   }

   return result;
}
