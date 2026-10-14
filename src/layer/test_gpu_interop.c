/* Real same-Intel device bridge interop fixture. Only this diagnostic maps a
 * staging buffer to inspect returned bytes; the importer never maps images. */
#define VK_USE_PLATFORM_WIN32_KHR
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <vulkan/vulkan.h>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "nr_gpu_bridge.h"
#include "nr_gpu_rpc.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s (Win32 %lu)\n", __LINE__, #x, (unsigned long)GetLastError()); ExitProcess(1); } } while (0)
#define VK_CHECK(x) do { VkResult result_ = (x); if (result_ != VK_SUCCESS) { fprintf(stderr, "FAIL line %d: %s = %d\n", __LINE__, #x, (int)result_); ExitProcess(1); } } while (0)

static void fill_shared(struct nr_gpu_bridge *b, VkQueue queue, uint32_t pattern)
{
    VK_CHECK(nr_gpu_bridge_begin(b, b->capture_command, b->capture_fence, &b->capture_pending));
    if (b->input_external) {
        VkBufferMemoryBarrier acquire = nr_gpu_bridge_buffer_barrier(b->input, b->bytes,
            VK_QUEUE_FAMILY_EXTERNAL, b->family, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        b->CmdPipelineBarrier(b->capture_command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &acquire, 0, NULL);
    }
    vkCmdFillBuffer(b->capture_command, b->input, 0, b->bytes, pattern);
    VkBufferMemoryBarrier release[2];
    release[0] = nr_gpu_bridge_buffer_barrier(b->input, b->bytes, b->family,
        VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    uint32_t count = 1;
    if (!b->output_external) release[count++] = nr_gpu_bridge_buffer_barrier(b->output, b->bytes,
        b->family, VK_QUEUE_FAMILY_EXTERNAL, 0, 0);
    b->CmdPipelineBarrier(b->capture_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, count, release, 0, NULL);
    VK_CHECK(b->EndCommandBuffer(b->capture_command));
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
        .pCommandBuffers = &b->capture_command, .signalSemaphoreCount = b->fenced ? 0 : 1,
        .pSignalSemaphores = b->fenced ? NULL : &b->ready};
    VK_CHECK(b->QueueSubmit(queue, 1, &submit, b->capture_fence));
    b->capture_pending = b->input_external = b->output_external = 1;
    if (b->fenced) VK_CHECK(nr_gpu_bridge_wait(b, b->capture_fence, &b->capture_pending));
}

static void verify_shared(struct nr_gpu_bridge *b, VkQueue queue, VkBuffer staging,
                           const uint32_t *mapped, uint32_t pattern)
{
    VK_CHECK(nr_gpu_bridge_begin(b, b->output_command, b->output_fence, &b->output_pending));
    VkBufferMemoryBarrier acquire = nr_gpu_bridge_buffer_barrier(b->output, b->bytes,
        VK_QUEUE_FAMILY_EXTERNAL, b->family, 0, VK_ACCESS_TRANSFER_READ_BIT);
    b->CmdPipelineBarrier(b->output_command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &acquire, 0, NULL);
    VkBufferCopy region = {.size = b->bytes};
    vkCmdCopyBuffer(b->output_command, b->output, staging, 1, &region);
    VkBufferMemoryBarrier release = nr_gpu_bridge_buffer_barrier(b->output, b->bytes,
        b->family, VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_READ_BIT, 0);
    b->CmdPipelineBarrier(b->output_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 1, &release, 0, NULL);
    VkBufferMemoryBarrier host = nr_gpu_bridge_buffer_barrier(staging, b->bytes,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    b->CmdPipelineBarrier(b->output_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &host, 0, NULL);
    VK_CHECK(b->EndCommandBuffer(b->output_command));
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
        .pCommandBuffers = &b->output_command, .waitSemaphoreCount = b->fenced ? 0 : 1,
        .pWaitSemaphores = b->fenced ? NULL : &b->done, .pWaitDstStageMask = b->fenced ? NULL : &stage};
    VK_CHECK(b->QueueSubmit(queue, 1, &submit, b->output_fence));
    b->output_pending = 1;
    VK_CHECK(nr_gpu_bridge_wait(b, b->output_fence, &b->output_pending));
    for (VkDeviceSize word = 0; word < b->bytes / sizeof *mapped; ++word) {
        if (mapped[word] != pattern) {
            fprintf(stderr, "returned word %llu = %08x, expected %08x\n",
                    (unsigned long long)word, mapped[word], pattern);
            ExitProcess(1);
        }
    }
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "NR GPU interop fixture", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    CHECK(count && count <= 32);
    VkPhysicalDevice physicals[32];
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, physicals));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties;
    for (uint32_t i = 0; i < count; ++i) {
        vkGetPhysicalDeviceProperties(physicals[i], &properties);
        if (properties.vendorID == 0x8086u) { physical = physicals[i]; break; }
    }
    CHECK(physical);
    VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
    vkGetPhysicalDeviceProperties2(physical, &props2);
    fprintf(stderr, "Device %s vendor %x device %x api %u.%u.%u driverVersion %u driverID %u driverName %s driverInfo %s\n",
            props2.properties.deviceName, props2.properties.vendorID, props2.properties.deviceID,
            VK_API_VERSION_MAJOR(props2.properties.apiVersion), VK_API_VERSION_MINOR(props2.properties.apiVersion),
            VK_API_VERSION_PATCH(props2.properties.apiVersion), props2.properties.driverVersion,
            driver.driverID, driver.driverName, driver.driverInfo);
    const VkExternalSemaphoreHandleTypeFlagBits handle_types[] = {
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT};
    for (unsigned timeline = 0; timeline <= 1; ++timeline) {
        for (unsigned hi = 0; hi < sizeof handle_types / sizeof handle_types[0]; ++hi) {
            VkSemaphoreTypeCreateInfo type = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY};
            VkPhysicalDeviceExternalSemaphoreInfo query = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
                .pNext = &type, .handleType = handle_types[hi]};
            VkExternalSemaphoreProperties capability = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
            vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &query, &capability);
            fprintf(stderr, "Semaphore %s handle=%x features=%x compatible=%x exportFromImported=%x\n",
                    timeline ? "timeline" : "binary", handle_types[hi], capability.externalSemaphoreFeatures,
                    capability.compatibleHandleTypes, capability.exportFromImportedHandleTypes);
        }
    }
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, NULL);
    CHECK(families && families <= 64);
    VkQueueFamilyProperties queues[64];
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, queues);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < families; ++i)
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    CHECK(family != UINT32_MAX);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    struct nr_gpu_device_patch patch;
    CHECK(nr_gpu_patch_device(&patch, physical, &dci, vkEnumerateDeviceExtensionProperties,
                              vkGetPhysicalDeviceFeatures2, vkGetPhysicalDeviceProperties));
    VkDevice device;
    VK_CHECK(vkCreateDevice(physical, &patch.info, NULL, &device));
    nr_gpu_patch_free(&patch);
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    struct nr_gpu_bridge b;
    VK_CHECK(nr_gpu_bridge_init(&b, device, physical, family, vkGetDeviceProcAddr,
        vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceExternalBufferProperties,
        vkGetPhysicalDeviceExternalSemaphoreProperties, vkGetPhysicalDeviceMemoryProperties,
        vkGetPhysicalDeviceQueueFamilyProperties));
    for (unsigned session = 1; session <= 2; ++session) {
        struct nr_gpu_packet packet;
        struct nr_gpu_reply reply;
        uint32_t width = session == 1 ? 64 : 96, height = session == 1 ? 64 : 80;
        VK_CHECK(nr_gpu_bridge_configure(&b, width, height, VK_FORMAT_B8G8R8A8_UNORM,
                                        UINT64_C(0x1357246810203040) + session, &packet));
        fprintf(stderr, "exported pid %u, handles %llx/%llx/%llx/%llx, bytes %llu, types %u/%u\n",
                packet.pid, (unsigned long long)packet.input_memory, (unsigned long long)packet.output_memory,
                (unsigned long long)packet.ready_sem, (unsigned long long)packet.done_sem,
                (unsigned long long)packet.bytes, packet.input_memory_type, packet.output_memory_type);
        DWORD peer = 0;
        CHECK(nr_gpu_rpc_peer(argv[1], &packet, &reply, &peer) && reply.status == NR_GPU_SUCCESS);
        CHECK(peer && reply.detail == peer);
        VkBufferCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = b.bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VkBuffer staging;
        VK_CHECK(vkCreateBuffer(device, &sci, NULL, &staging));
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, staging, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < b.memory_properties.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i))
                && (b.memory_properties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
        CHECK(type != UINT32_MAX);
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size, .memoryTypeIndex = type};
        VkDeviceMemory memory;
        VK_CHECK(vkAllocateMemory(device, &mai, NULL, &memory));
        VK_CHECK(vkBindBufferMemory(device, staging, memory, 0));
        void *mapped = NULL;
        VK_CHECK(vkMapMemory(device, memory, 0, b.bytes, 0, &mapped));
        for (unsigned frame = 0; frame < 3; ++frame) {
            uint32_t pattern = 0x12340000u | (session << 8) | frame;
            fill_shared(&b, queue, pattern);
            nr_gpu_bridge_packet(&b, NR_GPU_FRAME, &packet);
            CHECK(nr_gpu_rpc(argv[1], &packet, &reply) && reply.status == NR_GPU_SUCCESS);
            verify_shared(&b, queue, staging, mapped, pattern);
        }
        nr_gpu_bridge_packet(&b, NR_GPU_RELEASE, &packet);
        CHECK(nr_gpu_rpc(argv[1], &packet, &reply) && reply.status == NR_GPU_SUCCESS);
        VK_CHECK(nr_gpu_bridge_close(&b, NULL));
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, staging, NULL);
        vkFreeMemory(device, memory, NULL);
        printf("Interop %ux%u: three frames exact, protocol %u, exporter memory types %u/%u, peer %lu, RELEASE acknowledged\n",
               width, height, packet.version, packet.input_memory_type, packet.output_memory_type, (unsigned long)peer);
    }
    nr_gpu_rpc_reap();
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    puts("GPU Win32 interop: device/driver identity, dedicated memory, ownership synchronization, repeated frames and resize cleanup OK");
    return 0;
}
