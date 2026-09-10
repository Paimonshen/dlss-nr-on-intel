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
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/time.h>
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
	unsigned char *result;          /* the processed frame, held while the trigger is up */
	VkDeviceSize result_size;
	int holding;
	/* One frame captured before the one being processed, so the daemon can be told
	 * which pixels did not move. Filled on the present after the trigger goes up and
	 * released when it goes down; nothing is copied while the trigger is down, so a
	 * game that never triggers pays nothing for this. */
	unsigned char *earlier;
	VkDeviceSize earlier_size;
	int have_earlier;
	unsigned char *outgoing;        /* colour followed by the mask, for one send */
	VkDeviceSize outgoing_size;
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
static const char *socket_path;
static const char *trigger_path;
static int ui_mask;

/* The frame goes to a daemon over a Unix socket rather than being processed in
 * process: the implementation is Python and this is a shared object living inside the
 * game. For a photo mode the game is meant to stall anyway, so the round trip is free;
 * a per-frame pass would need the graph ported to C. */
static int exchange(const void *header, size_t header_size, const void *payload,
		    size_t payload_size, void *reply)
{
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) return -1;
	struct timeval timeout = { .tv_sec = 60 };
	if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout) ||
	    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout)) {
		close(fd); return -1;
	}
	struct sockaddr_un address = { .sun_family = AF_UNIX };
	snprintf(address.sun_path, sizeof address.sun_path, "%s", socket_path);
	if (connect(fd, (struct sockaddr *)&address, sizeof address) < 0) {
		fprintf(stderr, "[nr_layer] no daemon at %s\n", socket_path);
		close(fd);
		return -1;
	}
	const unsigned char *out = header;
	for (size_t sent = 0; sent < header_size; ) {
		ssize_t n = send(fd, out + sent, header_size - sent, MSG_NOSIGNAL);
		if (n < 0 && errno == EINTR) continue;
		if (n <= 0) { close(fd); return -1; }
		sent += (size_t)n;
	}
	out = payload;
	for (size_t sent = 0; sent < payload_size; ) {
		ssize_t n = send(fd, out + sent, payload_size - sent, MSG_NOSIGNAL);
		if (n < 0 && errno == EINTR) continue;
		if (n <= 0) { close(fd); return -1; }
		sent += (size_t)n;
	}
	unsigned char *in = reply;
	for (size_t got = 0; got < payload_size; ) {
		ssize_t n = read(fd, in + got, payload_size - got);
		if (n < 0 && errno == EINTR) continue;
		if (n <= 0) { close(fd); return -1; }
		got += (size_t)n;
	}
	close(fd);
	return 0;
}

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
		socket_path = getenv("NR_LAYER_SOCKET");
		trigger_path = getenv("NR_LAYER_TRIGGER");
		ui_mask = getenv("NR_LAYER_UI_MASK") != NULL;
		const char *every = getenv("NR_LAYER_EVERY");
		capture_every = every ? strtol(every, NULL, 10) : 0;
		fprintf(stderr, "[nr_layer] active; socket=%s trigger=%s capture=%s every=%ld\n",
			socket_path ? socket_path : "(none)",
			trigger_path ? trigger_path : "(none)",
			capture_path ? capture_path : "(off)", capture_every);
		if (ui_mask)
			fprintf(stderr, "[nr_layer] ui mask on: the first present after the "
				"trigger is kept to find what held still\n");
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
	patched.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

/* Move the presented frame between the swapchain image and a host-visible buffer.
 *
 * `vkQueueWaitIdle` around the transfer is the blunt way to know the frame is
 * finished: the proper route waits on the present's own semaphores, which means taking
 * them over from the application. For a photo mode the game is meant to stall anyway,
 * so the stall is the point rather than a cost. It has to change before the pass runs
 * every frame.
 */
static int ensure_resources(struct device_data *data, VkDeviceSize needed)
{
	if (!data->pool) {
		VkCommandPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = data->queue_family };
		PFN_vkCreateCommandPool create = (PFN_vkCreateCommandPool)
			data->get_device_proc(data->device, "vkCreateCommandPool");
		if (create(data->device, &info, NULL, &data->pool) != VK_SUCCESS) return -1;
	}
	if (data->staging_size >= needed) return 0;

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
				    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
					     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
	if (create_buffer(data->device, &info, NULL, &data->staging) != VK_SUCCESS) return -1;
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
	VkMemoryAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					    .allocationSize = mr.size, .memoryTypeIndex = type };
	if (allocate(data->device, &allocation, NULL, &data->staging_memory) != VK_SUCCESS)
		return -1;
	bind(data->device, data->staging, data->staging_memory, 0);
	map(data->device, data->staging_memory, 0, VK_WHOLE_SIZE, 0, &data->mapped);
	data->staging_size = needed;
	free(data->result);
	data->result = malloc((size_t)needed);
	data->result_size = needed;
	return data->result ? 0 : -1;
}

static int transfer(struct device_data *data, struct swapchain_data *chain, VkQueue queue,
		    uint32_t index, int to_image)
{
	PFN_vkAllocateCommandBuffers allocate_commands = (PFN_vkAllocateCommandBuffers)
		data->get_device_proc(data->device, "vkAllocateCommandBuffers");
	PFN_vkBeginCommandBuffer begin = (PFN_vkBeginCommandBuffer)
		data->get_device_proc(data->device, "vkBeginCommandBuffer");
	PFN_vkCmdPipelineBarrier barrier = (PFN_vkCmdPipelineBarrier)
		data->get_device_proc(data->device, "vkCmdPipelineBarrier");
	PFN_vkCmdCopyImageToBuffer copy_out = (PFN_vkCmdCopyImageToBuffer)
		data->get_device_proc(data->device, "vkCmdCopyImageToBuffer");
	PFN_vkCmdCopyBufferToImage copy_in = (PFN_vkCmdCopyBufferToImage)
		data->get_device_proc(data->device, "vkCmdCopyBufferToImage");
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

	VkImageLayout working = to_image ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
					 : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	VkImageMemoryBarrier into = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		.dstAccessMask = to_image ? VK_ACCESS_TRANSFER_WRITE_BIT
					  : VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.newLayout = working,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = chain->images[index],
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	barrier(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &into);

	VkBufferImageCopy region = {
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { chain->extent.width, chain->extent.height, 1 } };
	if (to_image)
		copy_in(commands, data->staging, chain->images[index], working, 1, &region);
	else
		copy_out(commands, chain->images[index], working, data->staging, 1, &region);

	VkImageMemoryBarrier back = into;
	back.srcAccessMask = into.dstAccessMask;
	back.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	back.oldLayout = working;
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
	return 0;
}

/* Which pixels are the interface.
 *
 * The shipped feature never has to ask: it inserts the pass before the interface is
 * drawn ("UI remains downstream"). A layer at `vkQueuePresentKHR` sees the composed
 * frame and has to work it out, and the one signal available is motion — an interface
 * holds still while the scene under it does not.
 *
 * That signal cannot separate an interface over a still scene from a still scene, so
 * when almost nothing moved the mask is refused rather than guessed: `settled` returns
 * 0 and the frame is sent unmasked, which is what a photo mode of a paused scene wants.
 */
static uint32_t settled(const unsigned char *now, const unsigned char *before,
			uint32_t pixels, unsigned char *mask)
{
	uint32_t held = 0;
	for (uint32_t i = 0; i < pixels; i++) {
		const unsigned char *a = now + 4 * i, *b = before + 4 * i;
		int moved = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) > 6;
		mask[i] = moved ? 0u : 0xFFu;
		held += !moved;
	}
	return held;
}

/* Whether a mask is worth sending at all. Over nine tenths held still means nothing
 * moved, so an interface cannot be told from a still scene and the mask would cover the
 * whole frame. Under a fiftieth means there is no interface worth a second plane on the
 * wire. Between them the signal is real. */
static int mask_worth_sending(uint32_t held, uint32_t pixels)
{
	return held < (uint32_t)((uint64_t)pixels * 9 / 10) && held > pixels / 50;
}

static int process_frame(struct device_data *data, struct swapchain_data *chain,
			 VkQueue queue, uint32_t index)
{
	VkDeviceSize needed = (VkDeviceSize)chain->extent.width * chain->extent.height * 4;
	if (ensure_resources(data, needed)) return -1;
	if (transfer(data, chain, queue, index, 0)) return -1;

	uint32_t pixels = chain->extent.width * chain->extent.height;
	int masked = 0;
	if (ui_mask && data->have_earlier && data->earlier_size >= needed) {
		if (data->outgoing_size < needed + pixels) {
			unsigned char *grown = realloc(data->outgoing, (size_t)needed + pixels);
			if (!grown) return -1;
			data->outgoing = grown;
			data->outgoing_size = needed + pixels;
		}
		memcpy(data->outgoing, data->mapped, (size_t)needed);
		uint32_t held = settled(data->mapped, data->earlier, pixels,
					data->outgoing + needed);
		masked = mask_worth_sending(held, pixels);
		fprintf(stderr, "[nr_layer] %u%% of the frame held still; ui mask %s\n",
			100u * held / pixels, masked ? "sent" : "refused");
	}

	uint32_t header[4] = { masked ? 0x314E524Eu : 0x304E524Eu, chain->extent.width,
			       chain->extent.height, (uint32_t)chain->format };
	if (capture_path) {
		FILE *file = fopen(capture_path, "wb");
		if (file) {
			fwrite(header, sizeof header, 1, file);
			fwrite(data->mapped, 1, (size_t)needed, file);
			fclose(file);
		}
	}
	if (!socket_path) return -1;
	/* The mask travels immediately after the colour, one byte a pixel. */
	if (masked) {
		if (exchange(header, sizeof header, data->outgoing,
			     (size_t)needed + pixels, data->result)) {
			fprintf(stderr, "[nr_layer] the daemon did not answer; frame unchanged\n");
			return -1;
		}
		memcpy(data->mapped, data->result, (size_t)needed);
		fprintf(stderr, "[nr_layer] processed %ux%u with a ui mask\n",
			chain->extent.width, chain->extent.height);
		return 0;
	}
	if (exchange(header, sizeof header, data->mapped, (size_t)needed, data->result)) {
		fprintf(stderr, "[nr_layer] the daemon did not answer; frame unchanged\n");
		return -1;
	}
	memcpy(data->mapped, data->result, (size_t)needed);
	fprintf(stderr, "[nr_layer] processed %ux%u\n", chain->extent.width,
		chain->extent.height);
	return 0;
}

VKAPI_ATTR VkResult VKAPI_CALL nr_QueuePresentKHR(VkQueue queue,
						  const VkPresentInfoKHR *info)
{
	frame_counter++;
	struct device_data *data = NULL;
	for (int i = 0; i < 8; i++) if (devices[i].device) { data = &devices[i]; break; }
	if (!data) return VK_ERROR_INITIALIZATION_FAILED;

	/* A file is the trigger, not a key: it works the same on X11 and Wayland, needs
	 * no input hooking inside another process's window, and can be set from a script
	 * or a hotkey daemon. While it exists the processed frame is held on screen. */
	int wanted = trigger_path && access(trigger_path, F_OK) == 0;
	if (!wanted && capture_every > 0)
		wanted = frame_counter % (unsigned long)capture_every == 0;

	for (uint32_t i = 0; i < info->swapchainCount; i++) {
		struct swapchain_data *chain = find_swapchain(info->pSwapchains[i]);
		if (!chain) continue;
		uint32_t index = info->pImageIndices[i];
		if (wanted && ui_mask && !data->holding && !data->have_earlier) {
			/* Spend the first present after the trigger keeping the frame, and
			 * process the next one against it. One frame of extra latency on a
			 * pass that already takes a second, and nothing at all while the
			 * trigger is down. */
			VkDeviceSize needed = (VkDeviceSize)chain->extent.width
					      * chain->extent.height * 4;
			if (ensure_resources(data, needed) == 0
			    && transfer(data, chain, queue, index, 0) == 0) {
				if (data->earlier_size < needed) {
					unsigned char *grown = realloc(data->earlier, (size_t)needed);
					if (grown) { data->earlier = grown; data->earlier_size = needed; }
				}
				if (data->earlier_size >= needed) {
					memcpy(data->earlier, data->mapped, (size_t)needed);
					data->have_earlier = 1;
				}
			}
		} else if (wanted && !data->holding) {
			if (process_frame(data, chain, queue, index) == 0) {
				data->holding = 1;
				transfer(data, chain, queue, index, 1);
			}
		} else if (wanted && data->holding) {
			memcpy(data->mapped, data->result, (size_t)data->result_size);
			transfer(data, chain, queue, index, 1);
		} else if (!wanted) {
			data->holding = 0;
			data->have_earlier = 0;
		}
	}
	return data->present(queue, info);
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
