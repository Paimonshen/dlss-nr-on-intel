/* Strict-policy host mocks. No Vulkan instance/device or GPU process exists. */
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>
#include <assert.h>
static unsigned host_reads, closes;
static BOOL tracked_close(HANDLE handle);
#define CloseHandle tracked_close
#define nr_link_read nr_link_read_real
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "nr_transport.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#undef nr_link_read
static int blocked_read(nr_link *link, void *data, size_t count)
{ (void)link; (void)data; (void)count; ++host_reads; assert(0 && "strict mode read legacy pixels"); return -1; }
#define nr_link_read blocked_read
#include "nr_layer.c"
#undef nr_link_read
#undef CloseHandle

static unsigned dispatches, presents, present_waits;
static unsigned char mapped_pixel[4], cached_pixel[4];
static VkSemaphore render_wait = (VkSemaphore)(uintptr_t)31;
static BOOL tracked_close(HANDLE handle) { ++closes; return CloseHandle(handle); }
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL forbidden_dispatch(VkDevice d, const char *name)
{ (void)d; (void)name; ++dispatches; assert(0 && "strict mode allocated/mapped/copied host pixels"); return NULL; }
static VKAPI_ATTR VkResult VKAPI_CALL original_present(VkQueue q, const VkPresentInfoKHR *info)
{
    (void)q;
    ++presents; present_waits = info->waitSemaphoreCount;
    assert(info->pWaitSemaphores == &render_wait && info->waitSemaphoreCount == 1);
    return VK_SUCCESS;
}
static struct device_data *fresh(void)
{
    memset(devices, 0, sizeof devices); memset(queues, 0, sizeof queues); memset(swapchains, 0, sizeof swapchains);
    dispatches = presents = present_waits = host_reads = closes = 0;
    struct device_data *d = &devices[0];
    d->device = (VkDevice)(uintptr_t)1;
    d->queue_family = 0; d->get_device_proc = forbidden_dispatch;
    d->present = original_present; d->queues_announced = 1;
    d->holding = 1; d->result_size = 4;
    memset(mapped_pixel, 2, sizeof mapped_pixel); memset(cached_pixel, 9, sizeof cached_pixel);
    d->mapped = mapped_pixel; d->result = cached_pixel;
    d->primary = (struct nr_surface_selection){8, 1, 1};
    queues[0] = (struct queue_data){(VkQueue)(uintptr_t)7, d->device, 0, 1};
    swapchains[0].swapchain = (VkSwapchainKHR)(uintptr_t)8; swapchains[0].device = d->device;
    swapchains[0].extent = (VkExtent2D){1, 1}; swapchains[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    swapchains[0].surface = (struct nr_surface_record){8, 0, 1, 1, 1};
    swapchains[0].images[0] = (VkImage)(uintptr_t)22; swapchains[0].image_count = 1;
    socket_path = "mock shared endpoint"; capture_path = trigger_path = NULL;
    live_every = 1; async_live = ui_mask = 0; capture_every = 0;
    return d;
}
static void present_original(void)
{
    uint32_t index = 0;
    VkPresentInfoKHR info = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_wait, .swapchainCount = 1,
        .pSwapchains = &swapchains[0].swapchain, .pImageIndices = &index};
    assert(nr_QueuePresentKHR(queues[0].queue, &info) == VK_SUCCESS);
    assert(presents == 1 && present_waits == 1 && !dispatches && !host_reads);
    assert(info.waitSemaphoreCount == 1 && !devices[0].waits_consumed && !devices[0].holding);
    for (unsigned i = 0; i < sizeof mapped_pixel; ++i) assert(mapped_pixel[i] == 2);
}
static void direct_guards(void)
{
    struct device_data *d = fresh();
    assert(ensure_resources(d, 4) == -1);
    assert(transfer(d, &swapchains[0], queues[0].queue, 0, 0) == -1);
    assert(transfer(d, &swapchains[0], queues[0].queue, 0, 1) == -1);
    assert(process_frame(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(process_frame_async(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(!dispatches && !host_reads && !d->waits_consumed);
}
int main(void)
{
    const char *value = getenv("NR_GPU_ONLY"); char *saved = value ? _strdup(value) : NULL;
    assert(!value || saved);
    _putenv_s("NR_GPU_ONLY", "");
    struct device_data *d = fresh(); assert(!gpu_only_mode() && host_pixel_path_allowed(d, "compatibility"));
    _putenv_s("NR_GPU_ONLY", "0"); assert(!gpu_only_mode());
    _putenv_s("NR_GPU_ONLY", "1"); assert(gpu_only_mode());
    direct_guards();
    fresh(); present_original(); /* shared transport disabled */
    d = fresh(); d->gpu_enabled = 1; d->gpu_blocked = 1; present_original(); /* failed transport */
    d = fresh(); d->gpu_enabled = 1; async_live = 1; present_original(); /* unsupported async */
    d = fresh(); live_every = 2; present_original(); /* no host holding between processed frames */
    d = fresh(); live_every = 0; capture_every = 1; ui_mask = 1; present_original(); /* photo/capture */
    d = fresh();
    HANDLE event = CreateEventW(NULL, FALSE, FALSE, NULL); assert(event);
    d->inflight = 1; d->inflight_fd = (int)(intptr_t)event; d->inflight_size = 8u * 1024u * 1024u;
    drop_inflight(d);
    assert(!d->inflight && closes == 1 && !host_reads);
    _putenv_s("NR_GPU_ONLY", saved ? saved : ""); free(saved);
    puts("GPU-only layer policy: direct host guards, original WSI waits, disabled/failed/unsupported modes, holding/photo and no legacy reply reads OK");
    return 0;
}
