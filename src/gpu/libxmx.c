/*
 * libxmx — a resident Vulkan compute context for the Xe2 XMX GEMM.
 *
 * The subprocess runner built a whole Vulkan instance, device and pipeline for every
 * matmul, which cost ~80 ms of fixed overhead per call and made any timing
 * meaningless. This keeps all of that alive across calls and reuses growable
 * host-visible buffers, so a dispatch costs a memcpy, a submit and a fence wait.
 *
 * Build: cc -O2 -shared -fPIC -I<vulkan headers> -o libxmx.so libxmx.c -lvulkan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define FAIL(msg, r) do { snprintf(g.err, sizeof g.err, "%s (%d)", msg, (int)(r)); return -1; } while (0)

struct buf { VkBuffer b; VkDeviceMemory m; void *p; VkDeviceSize cap; };

static struct {
	VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qi;
	VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe;
	VkDescriptorPool dpool; VkDescriptorSet set;
	VkCommandPool cpool; VkCommandBuffer cb; VkFence fence;
	struct buf A, B, C;
	char name[256]; char err[256]; int ready;
} g;

const char *xmx_error(void) { return g.err; }
const char *xmx_device(void) { return g.name; }

/* HOST_CACHED first, then anything host-visible.
 *
 * This machine offers memoryTypes[1] = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT and
 * memoryTypes[2] = the same plus HOST_CACHED. Taking the first match landed on the
 * uncached one, where reading the result back ran at ~80 MB/s and buried a 1.35
 * TFLOP/s kernel: a 147456x32x128 GEMM spent 1073 ms moving 85 MB. */
static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(g.pd, &mp);
	for (uint32_t pass = 0; pass < 2; pass++) {
		VkMemoryPropertyFlags need = want | (pass ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
		for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
			if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & need) == need)
				return i;
	}
	return UINT32_MAX;
}

static int ensure(struct buf *b, VkDeviceSize size)
{
	if (b->cap >= size)
		return 0;
	if (b->b) { vkUnmapMemory(g.dev, b->m); vkDestroyBuffer(g.dev, b->b, NULL); vkFreeMemory(g.dev, b->m, NULL); }
	VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
				  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
	VkResult r = vkCreateBuffer(g.dev, &bi, NULL, &b->b);
	if (r) FAIL("vkCreateBuffer", r);
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(g.dev, b->b, &mr);
	uint32_t mt = memtype(mr.memoryTypeBits,
			      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (mt == UINT32_MAX) FAIL("no host-visible memory type", 0);
	VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				    .allocationSize = mr.size, .memoryTypeIndex = mt };
	r = vkAllocateMemory(g.dev, &ai, NULL, &b->m);
	if (r) FAIL("vkAllocateMemory", r);
	vkBindBufferMemory(g.dev, b->b, b->m, 0);
	r = vkMapMemory(g.dev, b->m, 0, size, 0, &b->p);
	if (r) FAIL("vkMapMemory", r);
	b->cap = size;
	return 0;
}

int xmx_init(const char *spv_path)
{
	if (g.ready) return 0;
	VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
				  .pApplicationName = "libxmx", .apiVersion = VK_API_VERSION_1_3 };
	VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
	VkResult r = vkCreateInstance(&ici, NULL, &g.inst);
	if (r) FAIL("vkCreateInstance", r);

	uint32_t n = 0;
	vkEnumeratePhysicalDevices(g.inst, &n, NULL);
	if (!n) FAIL("no physical device", 0);
	VkPhysicalDevice *pds = calloc(n, sizeof *pds);
	vkEnumeratePhysicalDevices(g.inst, &n, pds);
	g.pd = pds[0];
	free(pds);
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(g.pd, &props);
	snprintf(g.name, sizeof g.name, "%s", props.deviceName);

	uint32_t nq = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, NULL);
	VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
	vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, qf);
	g.qi = UINT32_MAX;
	for (uint32_t i = 0; i < nq; i++)
		if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qi = i; break; }
	free(qf);
	if (g.qi == UINT32_MAX) FAIL("no compute queue", 0);

	VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .cooperativeMatrix = VK_TRUE };
	VkPhysicalDeviceVulkan12Features v12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &cm,
		.vulkanMemoryModel = VK_TRUE, .vulkanMemoryModelDeviceScope = VK_TRUE, .shaderFloat16 = VK_TRUE };
	VkPhysicalDeviceVulkan11Features v11 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &v12,
		.storageBuffer16BitAccess = VK_TRUE };
	VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v11 };
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
					.queueFamilyIndex = g.qi, .queueCount = 1, .pQueuePriorities = &prio };
	const char *ext[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
	VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
				   .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
				   .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext };
	r = vkCreateDevice(g.pd, &dci, NULL, &g.dev);
	if (r) FAIL("vkCreateDevice", r);
	vkGetDeviceQueue(g.dev, g.qi, 0, &g.q);

	VkDescriptorSetLayoutBinding bind[3];
	for (int i = 0; i < 3; i++)
		bind[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
							  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutCreateInfo dl = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					       .bindingCount = 3, .pBindings = bind };
	if ((r = vkCreateDescriptorSetLayout(g.dev, &dl, NULL, &g.dsl))) FAIL("dsl", r);
	VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 12 };
	VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
					   .setLayoutCount = 1, .pSetLayouts = &g.dsl,
					   .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
	if ((r = vkCreatePipelineLayout(g.dev, &pli, NULL, &g.pl))) FAIL("pipeline layout", r);

	FILE *f = fopen(spv_path, "rb");
	if (!f) FAIL("cannot open spv", 0);
	fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
	void *code = malloc(len);
	if (fread(code, 1, len, f) != (size_t)len) { fclose(f); FAIL("short spv read", 0); }
	fclose(f);
	VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
					 .codeSize = len, .pCode = code };
	VkShaderModule sm;
	if ((r = vkCreateShaderModule(g.dev, &smi, NULL, &sm))) FAIL("shader module", r);
	free(code);
	VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = g.pl };
	if ((r = vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &g.pipe))) FAIL("pipeline", r);
	vkDestroyShaderModule(g.dev, sm, NULL);

	VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
	VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					   .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
	if ((r = vkCreateDescriptorPool(g.dev, &dpi, NULL, &g.dpool))) FAIL("descriptor pool", r);
	VkDescriptorSetAllocateInfo dsa = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					    .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.dsl };
	if ((r = vkAllocateDescriptorSets(g.dev, &dsa, &g.set))) FAIL("descriptor set", r);

	VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
					 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
					 .queueFamilyIndex = g.qi };
	if ((r = vkCreateCommandPool(g.dev, &cpci, NULL, &g.cpool))) FAIL("command pool", r);
	VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
					    .commandPool = g.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
					    .commandBufferCount = 1 };
	if ((r = vkAllocateCommandBuffers(g.dev, &cba, &g.cb))) FAIL("command buffer", r);
	VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	if ((r = vkCreateFence(g.dev, &fi, NULL, &g.fence))) FAIL("fence", r);

	g.ready = 1;
	return 0;
}

/* Zero-copy path: hand the caller the mapped operand buffers so it can build A in
 * place and read C in place, instead of memcpying both across. On a shared-memory
 * APU those copies buy nothing. `xmx_reserve` may reallocate, so the pointers it
 * returns are valid only until the next call. */
int xmx_reserve(unsigned M, unsigned N, unsigned K, void **pa, void **pb, void **pc)
{
	if (!g.ready) FAIL("not initialised", 0);
	if (ensure(&g.A, (VkDeviceSize)M * K * 2) || ensure(&g.B, (VkDeviceSize)K * N * 2)
	    || ensure(&g.C, (VkDeviceSize)M * N * 4))
		return -1;
	if (pa) *pa = g.A.p;
	if (pb) *pb = g.B.p;
	if (pc) *pc = g.C.p;
	return 0;
}

/* iters > 1 dispatches the same work repeatedly inside one submit, so the caller can
 * time the GPU without host copies dominating. */
int xmx_gemm(unsigned M, unsigned N, unsigned K, const void *a, const void *b, void *c, unsigned iters)
{
	if (!g.ready) FAIL("not initialised", 0);
	VkDeviceSize sa = (VkDeviceSize)M * K * 2, sb = (VkDeviceSize)K * N * 2, sc = (VkDeviceSize)M * N * 4;
	if (ensure(&g.A, sa) || ensure(&g.B, sb) || ensure(&g.C, sc))
		return -1;
	VkDescriptorBufferInfo dbi[3] = { { g.A.b, 0, sa }, { g.B.b, 0, sb }, { g.C.b, 0, sc } };
	VkWriteDescriptorSet w[3];
	for (int i = 0; i < 3; i++)
		w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g.set,
					       .dstBinding = i, .descriptorCount = 1,
					       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
	vkUpdateDescriptorSets(g.dev, 3, w, 0, NULL);
	if (a) memcpy(g.A.p, a, sa);
	if (b) memcpy(g.B.p, b, sb);

	vkResetCommandBuffer(g.cb, 0);
	VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	vkBeginCommandBuffer(g.cb, &bi);
	vkCmdBindPipeline(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
	vkCmdBindDescriptorSets(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.set, 0, NULL);
	unsigned pc[3] = { M, N, K };
	vkCmdPushConstants(g.cb, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
	VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			       .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			       .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
	for (unsigned i = 0; i < (iters ? iters : 1); i++) {
		vkCmdDispatch(g.cb, N / 16, M / 8, 1);
		if (i + 1 < iters)
			vkCmdPipelineBarrier(g.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
	}
	vkEndCommandBuffer(g.cb);
	vkResetFences(g.dev, 1, &g.fence);
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g.cb };
	VkResult r = vkQueueSubmit(g.q, 1, &si, g.fence);
	if (r) FAIL("submit", r);
	r = vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, 60ull * 1000000000ull);
	if (r) FAIL("fence wait", r);
	if (c) memcpy(c, g.C.p, sc);
	return 0;
}
