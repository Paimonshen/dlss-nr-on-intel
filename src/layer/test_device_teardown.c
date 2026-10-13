/* CPU-only device teardown lifetime. Every Vulkan call is mocked; importing the
 * layer source does not create a Vulkan instance or touch a GPU. */
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define nr_gpu_rpc nr_gpu_rpc_real
#define nr_gpu_rpc_peer nr_gpu_rpc_peer_real
#include "nr_gpu_rpc.h"
#undef nr_gpu_rpc
#undef nr_gpu_rpc_peer
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
static int mock_gpu_rpc(const char *path, const struct nr_gpu_packet *packet, struct nr_gpu_reply *reply);
static int mock_gpu_rpc_peer(const char *path, const struct nr_gpu_packet *packet,
                             struct nr_gpu_reply *reply, DWORD *pid);
#define nr_gpu_rpc mock_gpu_rpc
#define nr_gpu_rpc_peer mock_gpu_rpc_peer
#endif
#include "nr_layer.c"
#ifdef _WIN32
#undef nr_gpu_rpc
#undef nr_gpu_rpc_peer
#endif
#include <assert.h>

#define H(type, value) ((type)(uintptr_t)(value))
static unsigned wait_calls, wait_fence_count, object_destroys, device_destroys, swapchain_destroys;
static VkResult wait_result;
static int pending_work, driver_lost, unavailable_wait;
#ifdef _WIN32
static unsigned frame_requests, release_requests;
static unsigned frame_status;
static int frame_answered, release_answered;
static struct device_data *rpc_device;
static int mock_gpu_rpc(const char *path, const struct nr_gpu_packet *packet, struct nr_gpu_reply *reply)
{
    (void)path;
    assert(rpc_device && rpc_device->gpu);
    memset(reply, 0, sizeof *reply);
    reply->session_low = (uint32_t)packet->session;
    reply->session_high = (uint32_t)(packet->session >> 32);
    if (packet->action == NR_GPU_FRAME) {
        ++frame_requests;
        reply->status = frame_status;
        return frame_answered;
    }
    assert(packet->action == NR_GPU_RELEASE);
    ++release_requests;
    reply->status = NR_GPU_SUCCESS;
    return release_answered;
}
static int mock_gpu_rpc_peer(const char *path, const struct nr_gpu_packet *packet,
                             struct nr_gpu_reply *reply, DWORD *pid)
{
    (void)path; (void)packet; (void)reply; (void)pid;
    assert(0 && "a negotiated bridge must not reopen HELLO during a deferred retry");
    return 0;
}
#endif

static void may_destroy(void) { assert(!pending_work || driver_lost); }
static VKAPI_ATTR VkResult VKAPI_CALL m_wait(VkDevice device, uint32_t count,
    const VkFence *fences, VkBool32 all, uint64_t timeout)
{
    (void)device;
    assert(count && count <= 4 && all && timeout == UINT64_C(10000000000));
    for (uint32_t i = 0; i < count; ++i) assert(fences[i]);
    ++wait_calls; wait_fence_count = count;
    if (wait_result == VK_SUCCESS) pending_work = 0;
    if (wait_result == VK_ERROR_DEVICE_LOST) { pending_work = 0; driver_lost = 1; }
    return wait_result;
}
static VKAPI_ATTR void VKAPI_CALL m_destroy_fence(VkDevice d, VkFence f, const VkAllocationCallbacks *a)
{ (void)d; (void)f; (void)a; may_destroy(); ++object_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_destroy_pool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a)
{ (void)d; (void)p; (void)a; may_destroy(); ++object_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_destroy_buffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a)
{ (void)d; (void)b; (void)a; may_destroy(); ++object_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_free_memory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a)
{ (void)d; (void)m; (void)a; may_destroy(); ++object_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_unmap(VkDevice d, VkDeviceMemory m)
{ (void)d; (void)m; may_destroy(); ++object_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_destroy_device(VkDevice d, const VkAllocationCallbacks *a)
{ (void)d; (void)a; may_destroy(); ++device_destroys; }
static VKAPI_ATTR void VKAPI_CALL m_destroy_swapchain(VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks *a)
{ (void)d; (void)s; (void)a; may_destroy(); ++swapchain_destroys; }
#ifdef _WIN32
static VKAPI_ATTR VkResult VKAPI_CALL m_reset_command(VkCommandBuffer c, VkCommandBufferResetFlags f)
{ (void)c; (void)f; assert(!pending_work); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL m_reset_fence(VkDevice d, uint32_t n, const VkFence *f)
{ (void)d; (void)f; assert(!pending_work && n == 1); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL m_begin(VkCommandBuffer c, const VkCommandBufferBeginInfo *i)
{ (void)c; (void)i; assert(!pending_work); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL m_end(VkCommandBuffer c)
{ (void)c; assert(!pending_work); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL m_barrier(VkCommandBuffer c, VkPipelineStageFlags src,
    VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t nm, const VkMemoryBarrier *m,
    uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *i)
{ (void)c; (void)src; (void)dst; (void)flags; (void)nm; (void)m; (void)nb; (void)b; (void)ni; (void)i; assert(!pending_work); }
static VKAPI_ATTR void VKAPI_CALL m_capture(VkCommandBuffer c, VkImage i, VkImageLayout l,
    VkBuffer b, uint32_t n, const VkBufferImageCopy *r)
{ (void)c; (void)i; (void)l; (void)b; assert(!pending_work && n == 1 && r->imageExtent.width); }
static VKAPI_ATTR VkResult VKAPI_CALL m_submit(VkQueue q, uint32_t n, const VkSubmitInfo *i, VkFence f)
{ (void)q; assert(!pending_work && n == 1 && i->commandBufferCount == 1 && f); pending_work = 1; return VK_SUCCESS; }
#endif
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL m_get(VkDevice device, const char *name)
{
    (void)device;
    if (!strcmp(name, "vkWaitForFences")) return unavailable_wait ? NULL : (PFN_vkVoidFunction)m_wait;
    if (!strcmp(name, "vkDestroyFence")) return (PFN_vkVoidFunction)m_destroy_fence;
    if (!strcmp(name, "vkDestroyCommandPool")) return (PFN_vkVoidFunction)m_destroy_pool;
    if (!strcmp(name, "vkDestroyBuffer")) return (PFN_vkVoidFunction)m_destroy_buffer;
    if (!strcmp(name, "vkFreeMemory")) return (PFN_vkVoidFunction)m_free_memory;
    if (!strcmp(name, "vkUnmapMemory")) return (PFN_vkVoidFunction)m_unmap;
    assert(0 && "unexpected teardown dispatch"); return NULL;
}
static struct device_data *fresh(int gpu_pending, int cpu_pending)
{
    memset(devices, 0, sizeof devices);
    memset(queues, 0, sizeof queues);
    memset(swapchains, 0, sizeof swapchains);
    wait_calls = wait_fence_count = object_destroys = device_destroys = swapchain_destroys = 0;
    pending_work = gpu_pending || cpu_pending;
    driver_lost = unavailable_wait = 0;
    wait_result = VK_SUCCESS;
    socket_path = NULL;
#ifdef _WIN32
    frame_requests = release_requests = 0;
    frame_status = NR_GPU_UNSUPPORTED; frame_answered = release_answered = 1;
    rpc_device = NULL;
#endif
    struct device_data *d = &devices[0];
    d->device = H(VkDevice, 1);
    d->get_device_proc = m_get;
    d->destroy_device = m_destroy_device;
    d->destroy_swapchain = m_destroy_swapchain;
    d->pool = H(VkCommandPool, 2);
    d->transfer_fence = H(VkFence, 3);
    d->staging = H(VkBuffer, 4);
    d->staging_memory = H(VkDeviceMemory, 5);
    d->mapped = (void *)(uintptr_t)6;
    d->result = malloc(16); d->earlier = malloc(16); d->outgoing = malloc(16);
    assert(d->result && d->earlier && d->outgoing);
    d->device_error = cpu_pending ? VK_TIMEOUT : VK_SUCCESS;
    queues[0].device = d->device; queues[0].queue = H(VkQueue, 7);
    swapchains[0].device = d->device; swapchains[0].swapchain = H(VkSwapchainKHR, 8);
    d->result_chain = swapchains[0].swapchain;
#ifdef _WIN32
    d->gpu = calloc(1, sizeof *d->gpu); assert(d->gpu);
    struct nr_gpu_bridge *b = d->gpu;
    b->initialized = b->resources = b->fenced = 1;
    b->device = d->device;
    b->capture_pending = b->output_pending = gpu_pending;
    b->capture_fence = H(VkFence, 11); b->output_fence = H(VkFence, 12);
    b->pool = H(VkCommandPool, 13);
    b->input = H(VkBuffer, 14); b->output = H(VkBuffer, 15);
    b->input_memory = H(VkDeviceMemory, 16); b->output_memory = H(VkDeviceMemory, 17);
    b->WaitForFences = m_wait;
    b->DestroyFence = m_destroy_fence; b->DestroyCommandPool = m_destroy_pool;
    b->DestroyBuffer = m_destroy_buffer; b->FreeMemory = m_free_memory;
#else
    assert(!gpu_pending);
#endif
    return d;
}
static void check_destroyed(struct device_data *d)
{
    assert(device_destroys == 1 && object_destroys && !d->device);
    assert(!queues[0].device && !swapchains[0].device);
}
static void test_completed_teardown(void)
{
    struct device_data *d = fresh(0, 0);
    nr_DestroyDevice(d->device, NULL);
    assert(!wait_calls); check_destroyed(d);
}
static void test_cpu_timeout_retains_everything(void)
{
    struct device_data *d = fresh(0, 1);
    VkDevice retained = d->device;
    unsigned char *held_result = d->result;
    wait_result = VK_TIMEOUT;
    nr_DestroyDevice(retained, NULL);
    assert(wait_calls == 1 && wait_fence_count == 1 && !device_destroys && !object_destroys);
    assert(d->device == retained && d->result == held_result && d->mapped && d->pool);
    assert(queues[0].device == retained && swapchains[0].device == retained);
    assert(d->device_error == VK_ERROR_DEVICE_LOST);
    wait_result = VK_SUCCESS;
    nr_DestroyDevice(retained, NULL);
    assert(wait_calls == 2); check_destroyed(d);
}
#ifdef _WIN32
static void test_synthetic_device_lost_is_not_driver_loss(void)
{
    struct device_data *d = fresh(1, 0);
    d->device_error = VK_ERROR_DEVICE_LOST;  /* layer-generated error after timeout */
    d->gpu->error = VK_TIMEOUT;
    struct nr_gpu_bridge *held_bridge = d->gpu;
    wait_result = VK_TIMEOUT;
    nr_DestroyDevice(d->device, NULL);
    assert(wait_calls == 1 && wait_fence_count == 3 && !device_destroys && !object_destroys);
    assert(d->gpu == held_bridge && d->gpu->capture_pending && d->gpu->output_pending && d->gpu_blocked);
    wait_result = VK_SUCCESS;
    nr_DestroyDevice(d->device, NULL);
    assert(wait_calls == 2); check_destroyed(d);
}
static void test_real_driver_loss_allows_cleanup(void)
{
    struct device_data *d = fresh(1, 1);
    wait_result = VK_ERROR_DEVICE_LOST;
    nr_DestroyDevice(d->device, NULL);
    assert(wait_calls == 1 && wait_fence_count == 3 && driver_lost); check_destroyed(d);
}
static void test_all_pending_fences_are_batched_once(void)
{
    struct device_data *d = fresh(1, 1);
    d->stalled_fence = H(VkFence, 18);
    nr_DestroyDevice(d->device, NULL);
    assert(wait_calls == 1 && wait_fence_count == 4); check_destroyed(d);
}
static void test_peer_unknown_alone_does_not_pin_drained_device(void)
{
    struct device_data *d = fresh(0, 0);
    d->gpu_exposed = 1;
    nr_DestroyDevice(d->device, NULL);
    assert(!wait_calls); check_destroyed(d);
}
static void test_unavailable_wait_never_destroys_inflight_device(void)
{
    struct device_data *d = fresh(1, 0);
    unavailable_wait = 1;
    nr_DestroyDevice(d->device, NULL);
    assert(!wait_calls && !device_destroys && !object_destroys && d->device && d->gpu);
    unavailable_wait = 0;
    nr_DestroyDevice(d->device, NULL);
    assert(wait_calls == 1); check_destroyed(d);
}
static void test_swapchain_timeout_retains_images_and_tracking(void)
{
    struct device_data *d = fresh(1, 0);
    VkSwapchainKHR chain = swapchains[0].swapchain;
    struct nr_gpu_bridge *held_bridge = d->gpu;
    wait_result = VK_TIMEOUT;
    nr_DestroySwapchainKHR(d->device, chain, NULL);
    assert(wait_calls == 1 && wait_fence_count == 2 && !swapchain_destroys && !object_destroys);
    assert(swapchains[0].swapchain == chain && d->result_chain == chain && d->gpu == held_bridge);
    assert(d->device_error == VK_ERROR_DEVICE_LOST);
    wait_result = VK_SUCCESS;
    nr_DestroySwapchainKHR(d->device, chain, NULL);
    assert(wait_calls == 2 && swapchain_destroys == 1 && !swapchains[0].swapchain && !d->result_chain);
    assert(d->device && !device_destroys);
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static void test_swapchain_real_driver_loss_permits_destruction(void)
{
    struct device_data *d = fresh(1, 0);
    VkSwapchainKHR chain = swapchains[0].swapchain;
    wait_result = VK_ERROR_DEVICE_LOST;
    nr_DestroySwapchainKHR(d->device, chain, NULL);
    assert(driver_lost && swapchain_destroys == 1 && !swapchains[0].swapchain && !d->result_chain);
    assert(d->device_error == VK_ERROR_DEVICE_LOST);
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static void test_drained_swapchain_is_safe_with_unknown_peer(void)
{
    struct device_data *d = fresh(0, 0);
    d->gpu_exposed = 1;
    nr_DestroySwapchainKHR(d->device, swapchains[0].swapchain, NULL);
    assert(!wait_calls && swapchain_destroys == 1 && !swapchains[0].swapchain && !d->result_chain);
    assert(d->gpu && !object_destroys && !device_destroys);
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static struct device_data *gpu_frame_fixture(void)
{
    struct device_data *d = fresh(0, 0);
    d->gpu_enabled = d->gpu_attached = d->gpu_exposed = 1;
    d->timing.configured = 1;
    d->frame_counter = 10;
    struct nr_gpu_bridge *b = d->gpu;
    b->fenced = 0;
    b->width = 1280; b->height = 720; b->format = VK_FORMAT_B8G8R8A8_UNORM;
    b->bytes = b->input_alloc = b->output_alloc = 1280u * 720u * 4u;
    b->session = 42;
    b->capture_command = H(VkCommandBuffer, 20);
    b->output_command = H(VkCommandBuffer, 21);
    b->ResetCommandBuffer = m_reset_command; b->ResetFences = m_reset_fence;
    b->BeginCommandBuffer = m_begin; b->EndCommandBuffer = m_end;
    b->CmdPipelineBarrier = m_barrier; b->CmdCopyImageToBuffer = m_capture;
    b->QueueSubmit = m_submit;
    swapchains[0].format = b->format;
    swapchains[0].extent = (VkExtent2D){1280, 720};
    swapchains[0].images[0] = H(VkImage, 22); swapchains[0].image_count = 1;
    rpc_device = d;
    socket_path = "mock control endpoint";
    live_every = 1; async_live = ui_mask = 0; capture_path = NULL;
    return d;
}
static void test_unsupported_frame_safely_releases_and_defers_retry(void)
{
    struct device_data *d = gpu_frame_fixture();
    assert(process_frame_gpu(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(frame_requests == 1 && release_requests == 1 && wait_calls == 1);
    assert(d->waits_consumed && !pending_work && !d->gpu && !d->device_error);
    assert(!d->gpu_blocked && d->gpu_retry_frame == 310);
    d->frame_counter++;
    assert(process_frame_gpu(d, &swapchains[0], queues[0].queue, 0) == 0);
    assert(frame_requests == 1 && release_requests == 1); /* Host path until retry. */
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static void test_failed_frame_rpc_remains_fatal_to_gpu_transport(void)
{
    struct device_data *d = gpu_frame_fixture();
    frame_answered = 0;
    assert(process_frame_gpu(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(d->gpu_blocked && !d->gpu && !d->gpu_retry_frame && !d->device_error);
    assert(frame_requests == 1 && release_requests == 1 && !pending_work);
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static void test_unsupported_frame_unsafe_release_retains_export(void)
{
    struct device_data *d = gpu_frame_fixture();
    release_answered = 0;
    assert(process_frame_gpu(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(d->gpu_blocked && d->gpu && !d->gpu_retry_frame && !d->device_error);
    assert(frame_requests == 1 && release_requests == 1 && !pending_work && !object_destroys);
    release_answered = 1;
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
static void test_unsupported_frame_failed_capture_drain_stays_pinned(void)
{
    struct device_data *d = gpu_frame_fixture();
    wait_result = VK_TIMEOUT;
    assert(process_frame_gpu(d, &swapchains[0], queues[0].queue, 0) == -1);
    assert(d->gpu_blocked && d->gpu && d->gpu->capture_pending && !d->gpu_retry_frame);
    assert(d->device_error == VK_ERROR_DEVICE_LOST && pending_work && !object_destroys);
    wait_result = VK_SUCCESS;
    nr_DestroyDevice(d->device, NULL); check_destroyed(d);
}
#endif
int main(void)
{
    test_completed_teardown(); test_cpu_timeout_retains_everything();
#ifdef _WIN32
    test_synthetic_device_lost_is_not_driver_loss(); test_real_driver_loss_allows_cleanup();
    test_all_pending_fences_are_batched_once(); test_peer_unknown_alone_does_not_pin_drained_device();
    test_unavailable_wait_never_destroys_inflight_device();
    test_swapchain_timeout_retains_images_and_tracking(); test_swapchain_real_driver_loss_permits_destruction();
    test_drained_swapchain_is_safe_with_unknown_peer();
    test_unsupported_frame_safely_releases_and_defers_retry(); test_failed_frame_rpc_remains_fatal_to_gpu_transport();
    test_unsupported_frame_unsafe_release_retains_export(); test_unsupported_frame_failed_capture_drain_stays_pinned();
#endif
    puts("Device teardown mock: one private-fence drain, timeout pinning, no tracking-slot reuse, real device loss and independent peer imports OK");
    return 0;
}
