/* CPU-only Vulkan dispatch mocks: cache lifetime and WSI wait ownership must
 * remain correct even when copies, fences or the device fail. No GPU is used. */
#include "nr_layer.c"
#include <assert.h>

static unsigned allocated, fences_created, commands_reset, fences_reset;
static unsigned submitted, waited, fences_destroyed, pools_destroyed, presented;
static unsigned expected_waits, observed_present_waits;
static unsigned loader_callbacks;
static int pending, terminal, loader_initialized;
static VkResult submit_result, wait_result, reset_result, loader_result;

static VKAPI_ATTR VkResult VKAPI_CALL mock_allocate(VkDevice d,
    const VkCommandBufferAllocateInfo *info, VkCommandBuffer *out)
{
    assert(info->commandPool && info->commandBufferCount == 1);
    allocated++;
    loader_initialized = 0;
    *out = (VkCommandBuffer)(uintptr_t)0x300;
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_set_loader_data(VkDevice device, void *object)
{
    assert(device == (VkDevice)(uintptr_t)0x100 && object == (void *)(uintptr_t)0x300);
    assert(!pending && allocated && !loader_initialized);
    ++loader_callbacks;
    if (loader_result == VK_SUCCESS) loader_initialized = 1;
    return loader_result;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_create_fence(VkDevice d,
    const VkFenceCreateInfo *info, const VkAllocationCallbacks *a, VkFence *out)
{
    fences_created++;
    *out = (VkFence)0x400;
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_reset_commands(VkCommandBuffer cb,
    VkCommandBufferResetFlags flags)
{
    assert(!pending && cb && loader_initialized);
    commands_reset++;
    return reset_result;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_reset_fences(VkDevice d, uint32_t count,
    const VkFence *fences)
{
    assert(!pending && count == 1 && fences[0]);
    fences_reset++;
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_begin(VkCommandBuffer cb,
    const VkCommandBufferBeginInfo *info)
{
    assert(!pending && cb && loader_initialized);
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_end(VkCommandBuffer cb)
{
    assert(!pending);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL mock_barrier(VkCommandBuffer cb,
    VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
    uint32_t mc, const VkMemoryBarrier *m, uint32_t bc, const VkBufferMemoryBarrier *b,
    uint32_t ic, const VkImageMemoryBarrier *i)
{
    assert(!pending);
}
static VKAPI_ATTR void VKAPI_CALL mock_capture(VkCommandBuffer cb, VkImage image,
    VkImageLayout layout, VkBuffer buffer, uint32_t count, const VkBufferImageCopy *region)
{
    assert(!pending && count == 1 && region->imageExtent.width);
}
static VKAPI_ATTR void VKAPI_CALL mock_return(VkCommandBuffer cb, VkBuffer buffer,
    VkImage image, VkImageLayout layout, uint32_t count, const VkBufferImageCopy *region)
{
    assert(!pending && count == 1 && region->imageExtent.height);
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_submit(VkQueue queue, uint32_t count,
    const VkSubmitInfo *info, VkFence fence)
{
    assert(!pending && count == 1 && info->commandBufferCount == 1 && fence);
    assert(info->waitSemaphoreCount == expected_waits);
    if (!expected_waits) assert(!info->pWaitSemaphores && !info->pWaitDstStageMask);
    for (unsigned i = 0; i < expected_waits; i++)
        assert(info->pWaitSemaphores[i] && info->pWaitDstStageMask[i] == VK_PIPELINE_STAGE_TRANSFER_BIT);
    submitted++;
    if (submit_result == VK_SUCCESS) pending = 1;
    return submit_result;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_wait(VkDevice d, uint32_t count,
    const VkFence *fences, VkBool32 all, uint64_t timeout)
{
    assert(pending && count == 1 && all == VK_TRUE && timeout == UINT64_MAX);
    waited++;
    if (wait_result == VK_SUCCESS) pending = 0;
    return wait_result;
}
static VKAPI_ATTR void VKAPI_CALL mock_destroy_fence(VkDevice d, VkFence fence,
    const VkAllocationCallbacks *a)
{
    assert(!pending || terminal);
    fences_destroyed++;
}
static VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice d, VkCommandPool pool,
    const VkAllocationCallbacks *a)
{
    assert(!pending || terminal);
    pools_destroyed++;
}
static VKAPI_ATTR VkResult VKAPI_CALL mock_present(VkQueue queue,
    const VkPresentInfoKHR *info)
{
    presented++;
    observed_present_waits = info->waitSemaphoreCount;
    return VK_SUCCESS;
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_dispatch(VkDevice d, const char *name)
{
#define MOCK(api, fn) if (!strcmp(name, "vk" #api)) return (PFN_vkVoidFunction)fn
    MOCK(AllocateCommandBuffers, mock_allocate);
    MOCK(CreateFence, mock_create_fence);
    MOCK(ResetCommandBuffer, mock_reset_commands);
    MOCK(ResetFences, mock_reset_fences);
    MOCK(BeginCommandBuffer, mock_begin);
    MOCK(EndCommandBuffer, mock_end);
    MOCK(CmdPipelineBarrier, mock_barrier);
    MOCK(CmdCopyImageToBuffer, mock_capture);
    MOCK(CmdCopyBufferToImage, mock_return);
    MOCK(QueueSubmit, mock_submit);
    MOCK(WaitForFences, mock_wait);
    MOCK(DestroyFence, mock_destroy_fence);
    MOCK(DestroyCommandPool, mock_destroy_pool);
#undef MOCK
    return NULL;
}

static VkSemaphore semaphores[] = { (VkSemaphore)0x501, (VkSemaphore)0x502, (VkSemaphore)0x503 };
static VkQueue queue = (VkQueue)(uintptr_t)0x600;
static struct swapchain_data chain = {
    .images = { (VkImage)0x700 }, .image_count = 1,
    .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { 1280, 720 }
};
static struct device_data *fresh(void)
{
    memset(devices, 0, sizeof devices);
    memset(queues, 0, sizeof queues);
    allocated = fences_created = commands_reset = fences_reset = 0;
    submitted = waited = fences_destroyed = pools_destroyed = presented = 0;
    expected_waits = 2; observed_present_waits = 99;
    pending = terminal = loader_initialized = 0;
    loader_callbacks = 0;
    submit_result = wait_result = reset_result = loader_result = VK_SUCCESS;
    struct device_data *data = &devices[0];
    data->device = (VkDevice)(uintptr_t)0x100;
    data->pool = (VkCommandPool)0x200;
    data->staging = (VkBuffer)0x800;
    data->get_device_proc = mock_dispatch;
    data->set_loader_data = mock_set_loader_data;
    data->present = mock_present;
    data->present_wait = semaphores;
    data->present_wait_count = expected_waits;
    return data;
}

static void test_reuse(void)
{
    struct device_data *data = fresh();
    VkPresentInfoKHR info = { .waitSemaphoreCount = 2, .pWaitSemaphores = semaphores };
    assert(transfer(data, &chain, queue, 0, 0) == 0);
    assert(loader_callbacks == 1 && loader_initialized);
    assert(data->waits_consumed && data->present_wait_count == 0);
    VkPipelineStageFlags *stages = data->transfer_stages;
    expected_waits = 0;
    assert(transfer(data, &chain, queue, 0, 1) == 0);
    assert(allocated == 1 && fences_created == 1 && commands_reset == 1 && fences_reset == 1);
    assert(loader_callbacks == 1); /* Cached command buffer keeps its loader dispatch. */
    assert(data->transfer_stages == stages && !fences_destroyed && !pools_destroyed);
    assert(present_now(data, queue, &info) == VK_SUCCESS && observed_present_waits == 0);
    assert(info.waitSemaphoreCount == 2); /* The game's original is untouched. */

    data->present_wait_count = expected_waits = 3;
    data->waits_consumed = 0;
    chain.extent.width = 1920; chain.extent.height = 1080;
    assert(transfer(data, &chain, queue, 0, 0) == 0);
    assert(allocated == 1 && fences_created == 1 && data->transfer_stage_capacity == 3);
    assert(loader_callbacks == 1);
    release_transfer_resources(data);
    assert(fences_destroyed == 1 && pools_destroyed == 1);
    assert(!data->transfer_commands && !data->transfer_fence && !data->transfer_stages);
}

static void test_failed_wait(VkResult failure)
{
    struct device_data *data = fresh();
    wait_result = failure;
    VkPresentInfoKHR info = { .waitSemaphoreCount = 2, .pWaitSemaphores = semaphores };
    assert(transfer(data, &chain, queue, 0, 0) == -1);
    assert(data->device_error == failure && data->waits_consumed && data->present_wait_count == 0);
    assert(data->transfer_commands && data->transfer_fence);
    assert(transfer(data, &chain, queue, 0, 1) == -1);
    assert(!commands_reset && !fences_reset && submitted == 1 && waited == 1);
    assert(present_now(data, queue, &info) == failure && !presented);
    assert(!fences_destroyed && !pools_destroyed);
    terminal = 1; /* Teardown of the terminal device, not a retry. */
    release_transfer_resources(data);
    assert(fences_destroyed == 1 && pools_destroyed == 1);
}

static void test_failed_submit(void)
{
    struct device_data *data = fresh();
    submit_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    assert(transfer(data, &chain, queue, 0, 0) == -1);
    assert(!data->waits_consumed && data->present_wait_count == 2 && !waited);
    assert(data->transfer_error == VK_ERROR_OUT_OF_HOST_MEMORY && !data->device_error);
    data->transfer_error = submit_result = VK_SUCCESS;
    assert(transfer(data, &chain, queue, 0, 0) == 0);
    assert(allocated == 1 && fences_created == 1 && commands_reset == 1 && fences_reset == 1);
    release_transfer_resources(data);

    data = fresh(); submit_result = VK_ERROR_DEVICE_LOST;
    assert(transfer(data, &chain, queue, 0, 0) == -1);
    assert(data->device_error == VK_ERROR_DEVICE_LOST && !data->waits_consumed);
    assert(transfer(data, &chain, queue, 0, 0) == -1 && submitted == 1);
    release_transfer_resources(data);
}

static void test_reset_failure_and_family_change(void)
{
    struct device_data *data = fresh();
    assert(transfer(data, &chain, queue, 0, 0) == 0);
    data->present_wait_count = expected_waits = 2; data->waits_consumed = 0;
    reset_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    assert(transfer(data, &chain, queue, 0, 1) == -1);
    assert(submitted == 1 && !data->waits_consumed && data->present_wait_count == 2);
    release_transfer_resources(data);

    data = fresh();
    assert(transfer(data, &chain, queue, 0, 0) == 0);
    queues[0].queue = queue; queues[0].device = data->device;
    queues[0].family = 3; queues[0].capture_ok = 1;
    data->queues_announced = 1;
    VkPresentInfoKHR info = { .waitSemaphoreCount = 2, .pWaitSemaphores = semaphores };
    assert(present_locked(queue, &info) == VK_SUCCESS);
    assert(data->queue_family == 3 && !data->pool && !data->transfer_commands);
    assert(fences_destroyed == 1 && pools_destroyed == 1 && observed_present_waits == 2);
}

static void test_loader_callback_failure_preserves_waits(void)
{
    struct device_data *data = fresh();
    loader_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    assert(transfer(data, &chain, queue, 0, 0) == -1);
    assert(allocated == 1 && loader_callbacks == 1 && !loader_initialized);
    assert(!fences_created && !submitted && !waited && !commands_reset && !fences_reset);
    assert(!data->transfer_commands && !data->transfer_fence);
    assert(!data->waits_consumed && data->present_wait_count == 2);
    assert(data->transfer_error == VK_ERROR_OUT_OF_HOST_MEMORY && !data->device_error);
    release_transfer_resources(data); /* The pool reclaims the unsubmitted allocation. */
    assert(pools_destroyed == 1 && !data->pool);
}

static void test_statistics(void)
{
    struct nr_layer_timing stats = { .configured = 1, .enabled = 1, .every = 2 };
    nr_timing_begin(&stats, 1, 2, 1280, 720, 0, 1, 0, 0);
    assert(stats.active);
    nr_timing_add(&stats, NR_TIME_CAPTURE, 1000);
    nr_timing_add(&stats, NR_TIME_CAPTURE, 2000);
    nr_timing_finish(&stats, 1);
    nr_timing_begin(&stats, 1, 2, 1280, 720, 0, 2, 0, 0);
    assert(!stats.active);
    nr_timing_finish(&stats, 0);
    nr_timing_begin(&stats, 1, 2, 1280, 720, 0, 3, 0, 0);
    nr_timing_add(&stats, NR_TIME_CAPTURE, 9000);
    nr_timing_finish(&stats, 1);
    unsigned n;
    assert(nr_timing_percentile(&stats, NR_TIME_CAPTURE, 50, &n) == 3000 && n == 2);
    assert(nr_timing_percentile(&stats, NR_TIME_CAPTURE, 95, NULL) == 9000);
    assert(nr_timing_percentile(&stats, NR_TIME_WRITEBACK_WAIT, 95, &n) == 0 && n == 0);
    assert(stats.requests == 3 && stats.delivered == 2 && stats.failed == 1);
    nr_timing_begin(&stats, 1, 2, 1920, 1080, 0, 4, 0, 0);
    assert(stats.count == 0 && stats.requests == 1 && stats.first_frame == 4 && stats.active);
    nr_timing_finish(&stats, 1);
    stats.every = 1;
    for (unsigned i = 0; i < NR_TIMING_WINDOW + 20; i++) {
        nr_timing_begin(&stats, 1, 2, 1920, 1080, 0, i + 5, 0, 0);
        nr_timing_add(&stats, NR_TIME_CAPTURE, i);
        nr_timing_finish(&stats, 1);
    }
    assert(stats.count == NR_TIMING_WINDOW);
    assert(nr_timing_percentile(&stats, NR_TIME_CAPTURE, 50, &n) == 83 && n == 128);
    struct nr_layer_timing disabled = { .configured = 1 };
    nr_timing_begin(&disabled, 1, 2, 1280, 720, 0, 1, 0, 0);
    nr_timing_finish(&disabled, 1);
    assert(!disabled.active && !disabled.requests && !disabled.count);
}

int main(void)
{
    test_reuse();
    test_failed_wait(VK_ERROR_DEVICE_LOST);
    test_failed_wait(VK_TIMEOUT);
    test_failed_submit();
    test_reset_failure_and_family_change();
    test_loader_callback_failure_preserves_waits();
    test_statistics();
    puts("transfer cache, terminal failures, WSI wait ownership and timing statistics pass (CPU mocks)");
    return 0;
}
