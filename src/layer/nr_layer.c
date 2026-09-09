/*
 * nr_layer — a Vulkan layer that hands the presented frame to DLSS-NR.
 *
 * Why a Vulkan layer, and not the route everyone else takes.
 *
 * Every published way of getting DLSS-NR into a game — OptiScaler's fork, the
 * ReShade bridges, the dual-GPU MGPU Bridge — loads NVIDIA's own nvngx_dlssnr.dll
 * and therefore needs an NVIDIA GPU (or, for the AMD lab, a PTX translation of it).
 * None of them can run here. What we have instead is a reimplementation that is
 * already Vulkan compute.
 *
 * And under Proton a DX12 game goes DX12 -> VKD3D-Proton -> Vulkan on ANV, which is
 * the same driver our shaders run on. So the frame we want is already a VkImage on
 * the device we already use: no D3D interop, no Windows DLL, no PCIe transfer. A
 * layer at vkQueuePresentKHR sees it. vkBasalt has done exactly this on Linux for
 * years, so the shape is proven.
 *
 * This file is the capture half: chain into the loader, force TRANSFER_SRC onto the
 * swapchain images, and copy the presented frame out on demand.
 *
 * Build: cc -O2 -shared -fPIC -o libnr_layer.so nr_layer.c -lvulkan
 */
#define VK_USE_PLATFORM_XLIB_KHR
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#define MAX_SWAPCHAINS 8
#define MAX_IMAGES 8

struct device_data {
	VkDevice device;
	VkPhysicalDevice physical;
	VkQueue queue;
	uint32_t queue_family;
	VkCommandPool pool;
	VkDeviceMemory staging_memory;
	VkBuffer staging;
	VkDeviceSize staging_size;
	void *mapped;
	PFN_vkGetDeviceProcAddr get_device_proc;
	PFN_vkQueuePresentKHR present;
	PFN_vkCreateSwapchainKHR create_swapchain;
	PFN_vkGetSwapchainImagesKHR get_swapchain_images;
	PFN_vkDestroyDevice destroy_device;
};

struct swapchain_data {
	VkSwapchainKHR swapchain;
	VkDevice device;
	VkImage images[MAX_IMAGES];
	uint32_t image_count;
	VkFormat format;
	VkExtent2D extent;
};

static struct device_data devices[8];
static struct swapchain_data swapchains[MAX_SWAPCHAINS];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static PFN_vkGetInstanceProcAddr next_instance_proc;
static VkInstance layer_instance;
static unsigned long frame_counter;
static const char *capture_path;
static long capture_every;

static struct device_data *find_device(VkDevice device)
{
	for (int i = 0; i < 8; i++)
		if (devices[i].device == device) return &devices[i];
	return NULL;
}

static struct swapchain_data *find_swapchain(VkSwapchainKHR swapchain)
{
	for (int i = 0; i < MAX_SWAPCHAINS; i++)
		if (swapchains[i].swapchain == swapchain) return &swapchains[i];
	return NULL;
}

/* The loader hands each layer a chain of get-proc-address functions; these walk it. */
static VkLayerInstanceCreateInfo *instance_chain(const VkInstanceCreateInfo *info)
{
	VkLayerInstanceCreateInfo *item = (VkLayerInstanceCreateInfo *)info->pNext;
	while (item && !(item->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
			 && item->function == VK_LAYER_LINK_INFO))
		item = (VkLayerInstanceCreateInfo *)item->pNext;
	return item;
}

static VkLayerDeviceCreateInfo *device_chain(const VkDeviceCreateInfo *info)
{
	VkLayerDeviceCreateInfo *item = (VkLayerDeviceCreateInfo *)info->pNext;
	while (item && !(item->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
			 && item->function == VK_LAYER_LINK_INFO))
		item = (VkLayerDeviceCreateInfo *)item->pNext;
	return item;
}

VKAPI_ATTR VkResult VKAPI_CALL nr_CreateInstance(const VkInstanceCreateInfo *info,
						 const VkAllocationCallbacks *allocator,
						 VkInstance *instance)
{
	VkLayerInstanceCreateInfo *link = instance_chain(info);
	if (!link) return VK_ERROR_INITIALIZATION_FAILED;
	next_instance_proc = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	PFN_vkCreateInstance create =
		(PFN_vkCreateInstance)next_instance_proc(NULL, "vkCreateInstance");
	VkResult r = create(info, allocator, instance);
	if (r == VK_SUCCESS) {
		/* Physical-device entry points cannot be resolved against a NULL
		 * instance — only the handful of global ones can — so the instance has
		 * to be kept. */
		layer_instance = *instance;
		capture_path = getenv("NR_LAYER_CAPTURE");
		const char *every = getenv("NR_LAYER_EVERY");
		capture_every = every ? strtol(every, NULL, 10) : 0;
		fprintf(stderr, "[nr_layer] active; capture=%s every=%ld\n",
			capture_path ? capture_path : "(off)", capture_every);
	}
	return r;
}

VKAPI_ATTR VkResult VKAPI_CALL nr_CreateDevice(VkPhysicalDevice physical,
					       const VkDeviceCreateInfo *info,
					       const VkAllocationCallbacks *allocator,
					       VkDevice *device)
{
	VkLayerDeviceCreateInfo *link = device_chain(info);
	if (!link) return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkGetInstanceProcAddr next_instance = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr next_device = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;

	PFN_vkCreateDevice create = (PFN_vkCreateDevice)next_instance(NULL, "vkCreateDevice");
	VkResult r = create(physical, info, allocator, device);
	if (r != VK_SUCCESS) return r;

	pthread_mutex_lock(&lock);
	struct device_data *data = NULL;
	for (int i = 0; i < 8; i++) if (!devices[i].device) { data = &devices[i]; break; }
	if (data) {
		memset(data, 0, sizeof *data);
		data->device = *device;
		data->physical = physical;
		data->get_device_proc = next_device;
		data->present = (PFN_vkQueuePresentKHR)next_device(*device, "vkQueuePresentKHR");
		data->create_swapchain =
			(PFN_vkCreateSwapchainKHR)next_device(*device, "vkCreateSwapchainKHR");
		data->get_swapchain_images =
			(PFN_vkGetSwapchainImagesKHR)next_device(*device, "vkGetSwapchainImagesKHR");
		data->destroy_device = (PFN_vkDestroyDevice)next_device(*device, "vkDestroyDevice");
		data->queue_family = info->queueCreateInfoCount
			? info->pQueueCreateInfos[0].queueFamilyIndex : 0;
	}
	pthread_mutex_unlock(&lock);
	return r;
}

/* The swapchain images are the frame we want, so they have to be copyable. The loader
 * lets a layer edit the create info on the way down; without TRANSFER_SRC the copy
 * below is invalid, and this is the one place it can be added. */
VKAPI_ATTR VkResult VKAPI_CALL nr_CreateSwapchainKHR(VkDevice device,
						     const VkSwapchainCreateInfoKHR *info,
						     const VkAllocationCallbacks *allocator,
						     VkSwapchainKHR *swapchain)
{
	struct device_data *data = find_device(device);
	if (!data) return VK_ERROR_INITIALIZATION_FAILED;
	VkSwapchainCreateInfoKHR patched = *info;
	patched.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	VkResult r = data->create_swapchain(device, &patched, allocator, swapchain);
	if (r != VK_SUCCESS)
		r = data->create_swapchain(device, info, allocator, swapchain);
	if (r != VK_SUCCESS) return r;

	pthread_mutex_lock(&lock);
	struct swapchain_data *entry = NULL;
	for (int i = 0; i < MAX_SWAPCHAINS; i++)
		if (!swapchains[i].swapchain) { entry = &swapchains[i]; break; }
	if (entry) {
		memset(entry, 0, sizeof *entry);
		entry->swapchain = *swapchain;
		entry->device = device;
		entry->format = info->imageFormat;
		entry->extent = info->imageExtent;
		entry->image_count = MAX_IMAGES;
		data->get_swapchain_images(device, *swapchain, &entry->image_count, entry->images);
		fprintf(stderr, "[nr_layer] swapchain %ux%u format %d, %u images\n",
			entry->extent.width, entry->extent.height, entry->format,
			entry->image_count);
	}
	pthread_mutex_unlock(&lock);
	return r;
}

static uint32_t memory_type(struct device_data *data, uint32_t bits,
			    VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties properties;
	PFN_vkGetPhysicalDeviceMemoryProperties get =
		(PFN_vkGetPhysicalDeviceMemoryProperties)next_instance_proc(
			layer_instance, "vkGetPhysicalDeviceMemoryProperties");
	if (!get) return UINT32_MAX;
	get(data->physical, &properties);
	for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
		if ((bits & (1u << i))
		    && (properties.memoryTypes[i].propertyFlags & want) == want)
			return i;
	return UINT32_MAX;
}

/* Copy the presented image into a host-visible buffer and write it out.
 *
 * `vkQueueWaitIdle` before the copy is the blunt way to be sure the frame is
 * finished: the proper route is to wait on the present's own semaphores, which means
 * taking them over from the application. For an on-demand capture the stall costs a
 * frame and nothing else, so the blunt way is the honest one until the pass runs
 * every frame. */
static int nr_layer_capture(struct device_data *data, struct swapchain_data *chain,
			    VkQueue queue, uint32_t index, const char *path)
{
	if (index >= chain->image_count) return -1;
	VkDeviceSize needed = (VkDeviceSize)chain->extent.width * chain->extent.height * 4;

	if (!data->pool) {
		VkCommandPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = data->queue_family };
		PFN_vkCreateCommandPool create = (PFN_vkCreateCommandPool)
			data->get_device_proc(data->device, "vkCreateCommandPool");
		if (create(data->device, &info, NULL, &data->pool) != VK_SUCCESS) return -1;
	}
	if (data->staging_size < needed) {
		PFN_vkCreateBuffer create_buffer = (PFN_vkCreateBuffer)
			data->get_device_proc(data->device, "vkCreateBuffer");
		PFN_vkGetBufferMemoryRequirements requirements = (PFN_vkGetBufferMemoryRequirements)
			data->get_device_proc(data->device, "vkGetBufferMemoryRequirements");
		PFN_vkAllocateMemory allocate = (PFN_vkAllocateMemory)
			data->get_device_proc(data->device, "vkAllocateMemory");
		PFN_vkBindBufferMemory bind = (PFN_vkBindBufferMemory)
			data->get_device_proc(data->device, "vkBindBufferMemory");
		PFN_vkMapMemory map = (PFN_vkMapMemory)
			data->get_device_proc(data->device, "vkMapMemory");
		VkBufferCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
					    .size = needed,
					    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
		if (create_buffer(data->device, &info, NULL, &data->staging) != VK_SUCCESS)
			return -1;
		VkMemoryRequirements mr;
		requirements(data->device, data->staging, &mr);
		uint32_t type = memory_type(data, mr.memoryTypeBits,
					    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
					    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
					    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
		if (type == UINT32_MAX)
			type = memory_type(data, mr.memoryTypeBits,
					   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
					   | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (type == UINT32_MAX) return -1;
		VkMemoryAllocateInfo allocation = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = mr.size, .memoryTypeIndex = type };
		if (allocate(data->device, &allocation, NULL, &data->staging_memory) != VK_SUCCESS)
			return -1;
		bind(data->device, data->staging, data->staging_memory, 0);
		map(data->device, data->staging_memory, 0, VK_WHOLE_SIZE, 0, &data->mapped);
		data->staging_size = needed;
	}

	PFN_vkAllocateCommandBuffers allocate_commands = (PFN_vkAllocateCommandBuffers)
		data->get_device_proc(data->device, "vkAllocateCommandBuffers");
	PFN_vkBeginCommandBuffer begin = (PFN_vkBeginCommandBuffer)
		data->get_device_proc(data->device, "vkBeginCommandBuffer");
	PFN_vkCmdPipelineBarrier barrier = (PFN_vkCmdPipelineBarrier)
		data->get_device_proc(data->device, "vkCmdPipelineBarrier");
	PFN_vkCmdCopyImageToBuffer copy = (PFN_vkCmdCopyImageToBuffer)
		data->get_device_proc(data->device, "vkCmdCopyImageToBuffer");
	PFN_vkEndCommandBuffer end = (PFN_vkEndCommandBuffer)
		data->get_device_proc(data->device, "vkEndCommandBuffer");
	PFN_vkQueueSubmit submit = (PFN_vkQueueSubmit)
		data->get_device_proc(data->device, "vkQueueSubmit");
	PFN_vkQueueWaitIdle wait = (PFN_vkQueueWaitIdle)
		data->get_device_proc(data->device, "vkQueueWaitIdle");
	PFN_vkFreeCommandBuffers free_commands = (PFN_vkFreeCommandBuffers)
		data->get_device_proc(data->device, "vkFreeCommandBuffers");

	VkCommandBuffer commands;
	VkCommandBufferAllocateInfo alloc = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = data->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1 };
	if (allocate_commands(data->device, &alloc, &commands) != VK_SUCCESS) return -1;
	VkCommandBufferBeginInfo beginning = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	begin(commands, &beginning);
	VkImageMemoryBarrier to_source = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = chain->images[index],
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	barrier(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_source);
	VkBufferImageCopy region = {
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { chain->extent.width, chain->extent.height, 1 } };
	copy(commands, chain->images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	     data->staging, 1, &region);
	VkImageMemoryBarrier back = to_source;
	back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	back.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	back.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &back);
	end(commands);

	wait(queue);
	VkSubmitInfo submission = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				    .commandBufferCount = 1, .pCommandBuffers = &commands };
	submit(queue, 1, &submission, VK_NULL_HANDLE);
	wait(queue);
	free_commands(data->device, data->pool, 1, &commands);

	FILE *file = fopen(path, "wb");
	if (!file) return -1;
	/* A header the Python side can read without guessing: magic, extent, format. */
	uint32_t header[4] = { 0x304E524Eu, chain->extent.width, chain->extent.height,
			       (uint32_t)chain->format };
	fwrite(header, sizeof header, 1, file);
	fwrite(data->mapped, 1, (size_t)needed, file);
	fclose(file);
	fprintf(stderr, "[nr_layer] captured frame %lu -> %s (%ux%u, format %d)\n",
		frame_counter, path, chain->extent.width, chain->extent.height,
		chain->format);
	return 0;
}

VKAPI_ATTR VkResult VKAPI_CALL nr_QueuePresentKHR(VkQueue queue,
						  const VkPresentInfoKHR *info)
{
	frame_counter++;
	struct device_data *data = NULL;
	for (int i = 0; i < 8; i++) if (devices[i].device) { data = &devices[i]; break; }
	if (data && capture_path && capture_every > 0
	    && frame_counter % (unsigned long)capture_every == 0) {
		for (uint32_t i = 0; i < info->swapchainCount; i++) {
			struct swapchain_data *chain = find_swapchain(info->pSwapchains[i]);
			if (chain)
				nr_layer_capture(data, chain, queue, info->pImageIndices[i],
						 capture_path);
		}
	}
	return data ? data->present(queue, info) : VK_ERROR_INITIALIZATION_FAILED;
}

#define INTERCEPT(name) if (!strcmp(pName, "vk" #name)) return (PFN_vkVoidFunction)nr_##name

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL nr_GetDeviceProcAddr(VkDevice device,
							      const char *pName)
{
	INTERCEPT(QueuePresentKHR);
	INTERCEPT(CreateSwapchainKHR);
	struct device_data *data = find_device(device);
	return data ? data->get_device_proc(device, pName) : NULL;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL nr_GetInstanceProcAddr(VkInstance instance,
								const char *pName)
{
	INTERCEPT(CreateInstance);
	INTERCEPT(CreateDevice);
	INTERCEPT(GetInstanceProcAddr);
	INTERCEPT(GetDeviceProcAddr);
	INTERCEPT(QueuePresentKHR);
	INTERCEPT(CreateSwapchainKHR);
	return next_instance_proc ? next_instance_proc(instance, pName) : NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(
	VkNegotiateLayerInterface *interface)
{
	if (interface->loaderLayerInterfaceVersion < 2)
		return VK_ERROR_INITIALIZATION_FAILED;
	interface->loaderLayerInterfaceVersion = 2;
	interface->pfnGetInstanceProcAddr = nr_GetInstanceProcAddr;
	interface->pfnGetDeviceProcAddr = nr_GetDeviceProcAddr;
	interface->pfnGetPhysicalDeviceProcAddr = NULL;
	return VK_SUCCESS;
}
