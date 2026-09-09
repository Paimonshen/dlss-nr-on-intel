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
	VkPipelineLayout plb; VkPipeline pipeb;
	VkDescriptorPool dpool; VkDescriptorSet set;
	VkCommandPool cpool; VkCommandBuffer cb; VkFence fence;
	struct buf A, B, C;
	/* resident path */
	VkPipelineLayout rpl; VkPipeline rgemm, rtiled, runary, rrow, rhistory;
	unsigned tiling, tilem, tilen;
	VkCommandBuffer rcb; VkFence rfence; int recording, recorded, rready;
	char name[256]; char err[256]; int ready;
} g;

/* Device-resident buffers. The graph's activations live here between blocks instead
 * of being read back to the host after every GEMM; on a shared-memory APU the mapping
 * is HOST_CACHED, so the host can still write inputs and read outputs in place. */
#define MAX_RBUF 8192
struct rbuf { VkBuffer b; VkDeviceMemory m; void *p; VkDeviceAddress addr; VkDeviceSize size; int live; };
static struct rbuf rbufs[MAX_RBUF];

struct push {
	uint64_t a, b, c, d;
	uint32_t m, n, k, batch, sa, sb, sc, flags;
	float p0, p1, p2, p3;
	uint32_t lda, ldb, ldc, spare;
};

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

static int build_pipeline(const char *spv_path, VkPipelineLayout layout, VkPipeline *out)
{
	FILE *f = fopen(spv_path, "rb");
	if (!f) FAIL("cannot open spv", 0);
	fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
	void *code = malloc(len);
	if (fread(code, 1, len, f) != (size_t)len) { fclose(f); free(code); FAIL("short spv read", 0); }
	fclose(f);
	VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
					 .codeSize = len, .pCode = code };
	VkShaderModule sm;
	VkResult r = vkCreateShaderModule(g.dev, &smi, NULL, &sm);
	free(code);
	if (r) FAIL("shader module", r);
	VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = layout };
	r = vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpi, NULL, out);
	vkDestroyShaderModule(g.dev, sm, NULL);
	if (r) FAIL("pipeline", r);
	return 0;
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
	/* bufferDeviceAddress lets the resident path pass operands as 64-bit pointers in
	 * push constants, so a whole block of dispatches records into one command buffer
	 * without a descriptor pool. scalarBlockLayout matches the shaders' layout. */
	VkPhysicalDeviceVulkan12Features v12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &cm,
		.vulkanMemoryModel = VK_TRUE, .vulkanMemoryModelDeviceScope = VK_TRUE, .shaderFloat16 = VK_TRUE,
		.bufferDeviceAddress = VK_TRUE, .scalarBlockLayout = VK_TRUE };
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

	if (build_pipeline(spv_path, g.pl, &g.pipe))
		return -1;

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


/* The batched pipeline is built on first use: the extra shader takes seven push
 * constants instead of three, so it needs its own pipeline layout. */
int xmx_init_batched(const char *spv_path)
{
	if (!g.ready) FAIL("not initialised", 0);
	if (g.pipeb) return 0;
	VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 28 };
	VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
					   .setLayoutCount = 1, .pSetLayouts = &g.dsl,
					   .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
	VkResult r = vkCreatePipelineLayout(g.dev, &pli, NULL, &g.plb);
	if (r) FAIL("batched pipeline layout", r);
	return build_pipeline(spv_path, g.plb, &g.pipeb);
}

/* Byte capacities, so the caller can lay the three tensors out itself. */
int xmx_reserve_bytes(unsigned long long a, unsigned long long b, unsigned long long c,
		      void **pa, void **pb, void **pc)
{
	if (!g.ready) FAIL("not initialised", 0);
	if (ensure(&g.A, a) || ensure(&g.B, b) || ensure(&g.C, c))
		return -1;
	if (pa) *pa = g.A.p;
	if (pb) *pb = g.B.p;
	if (pc) *pc = g.C.p;
	return 0;
}

int xmx_gemm_batched(unsigned M, unsigned N, unsigned K, unsigned batch,
		     unsigned sa, unsigned sb, unsigned sc, unsigned bt)
{
	if (!g.pipeb) FAIL("batched pipeline not built", 0);
	VkDeviceSize sza = (VkDeviceSize)batch * sa * 2, szb = (VkDeviceSize)batch * sb * 2,
		     szc = (VkDeviceSize)batch * sc * 4;
	VkDescriptorBufferInfo dbi[3] = { { g.A.b, 0, sza }, { g.B.b, 0, szb }, { g.C.b, 0, szc } };
	VkWriteDescriptorSet w[3];
	for (int i = 0; i < 3; i++)
		w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g.set,
					       .dstBinding = i, .descriptorCount = 1,
					       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
	vkUpdateDescriptorSets(g.dev, 3, w, 0, NULL);

	vkResetCommandBuffer(g.cb, 0);
	VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	vkBeginCommandBuffer(g.cb, &bi);
	vkCmdBindPipeline(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeb);
	vkCmdBindDescriptorSets(g.cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.plb, 0, 1, &g.set, 0, NULL);
	unsigned push[7] = { M, N, K, sa, sb, sc, bt };
	vkCmdPushConstants(g.cb, g.plb, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, push);
	vkCmdDispatch(g.cb, N / 16, M / 8, batch);
	vkEndCommandBuffer(g.cb);
	vkResetFences(g.dev, 1, &g.fence);
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g.cb };
	VkResult r = vkQueueSubmit(g.q, 1, &si, g.fence);
	if (r) FAIL("submit", r);
	r = vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, 60ull * 1000000000ull);
	if (r) FAIL("fence wait", r);
	return 0;
}

/* ------------------------------------------------------------------------- */
/* The resident runtime.                                                      */
/*                                                                            */
/* Operands are 64-bit device addresses in the push constants rather than      */
/* descriptor bindings, so recording is just push-and-dispatch and a whole     */
/* block's dispatches go into one command buffer with one fence at the end,    */
/* instead of one submit per GEMM. Activations stay in device buffers between  */
/* passes; the point is not a faster kernel but the traffic that disappears.   */
/* ------------------------------------------------------------------------- */

int xmx_res_init(const char *gemm_spv, const char *unary_spv, const char *row_spv,
		 const char *history_spv, const char *tiled_spv)
{
	if (!g.ready) FAIL("not initialised", 0);
	if (g.rready) return 0;
	VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(struct push) };
	VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
					   .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
	VkResult r = vkCreatePipelineLayout(g.dev, &pli, NULL, &g.rpl);
	if (r) FAIL("resident pipeline layout", r);
	if (build_pipeline(gemm_spv, g.rpl, &g.rgemm) || build_pipeline(unary_spv, g.rpl, &g.runary)
	    || build_pipeline(row_spv, g.rpl, &g.rrow)
	    || build_pipeline(history_spv, g.rpl, &g.rhistory)
	    || build_pipeline(tiled_spv, g.rpl, &g.rtiled))
		return -1;
	const char *tile = getenv("XMX_TILE_K");
	g.tiling = tile ? atoi(tile) : 1;
	const char *bm = getenv("XMX_TILE_M"), *bn = getenv("XMX_TILE_N");
	g.tilem = bm ? atoi(bm) : 16; g.tilen = bn ? atoi(bn) : 32;
	VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
					    .commandPool = g.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
					    .commandBufferCount = 1 };
	if ((r = vkAllocateCommandBuffers(g.dev, &cba, &g.rcb))) FAIL("resident command buffer", r);
	VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	if ((r = vkCreateFence(g.dev, &fi, NULL, &g.rfence))) FAIL("resident fence", r);
	g.rready = 1;
	return 0;
}

int xmx_buf_create(unsigned long long bytes)
{
	if (!g.rready) FAIL("resident runtime not initialised", 0);
	int id = -1;
	for (int i = 0; i < MAX_RBUF; i++)
		if (!rbufs[i].live) { id = i; break; }
	if (id < 0) FAIL("out of buffer slots", 0);
	struct rbuf *rb = &rbufs[id];
	VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes ? bytes : 4,
				  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
					   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	VkResult r = vkCreateBuffer(g.dev, &bi, NULL, &rb->b);
	if (r) FAIL("vkCreateBuffer (resident)", r);
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(g.dev, rb->b, &mr);
	uint32_t mt = memtype(mr.memoryTypeBits,
			      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (mt == UINT32_MAX) FAIL("no host-visible memory type", 0);
	VkMemoryAllocateFlagsInfo fl = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
					 .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
	VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fl,
				    .allocationSize = mr.size, .memoryTypeIndex = mt };
	if ((r = vkAllocateMemory(g.dev, &ai, NULL, &rb->m))) FAIL("vkAllocateMemory (resident)", r);
	vkBindBufferMemory(g.dev, rb->b, rb->m, 0);
	if ((r = vkMapMemory(g.dev, rb->m, 0, VK_WHOLE_SIZE, 0, &rb->p))) FAIL("vkMapMemory (resident)", r);
	VkBufferDeviceAddressInfo ai2 = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = rb->b };
	rb->addr = vkGetBufferDeviceAddress(g.dev, &ai2);
	rb->size = bi.size;
	rb->live = 1;
	return id;
}

void *xmx_buf_ptr(int id)
{
	return (id >= 0 && id < MAX_RBUF && rbufs[id].live) ? rbufs[id].p : NULL;
}

unsigned long long xmx_buf_bytes(int id)
{
	return (id >= 0 && id < MAX_RBUF && rbufs[id].live) ? (unsigned long long)rbufs[id].size : 0;
}

int xmx_buf_destroy(int id)
{
	if (id < 0 || id >= MAX_RBUF || !rbufs[id].live) return 0;
	vkUnmapMemory(g.dev, rbufs[id].m);
	vkDestroyBuffer(g.dev, rbufs[id].b, NULL);
	vkFreeMemory(g.dev, rbufs[id].m, NULL);
	rbufs[id] = (struct rbuf){ 0 };
	return 0;
}

static VkDeviceAddress addr_of(int id)
{
	return (id >= 0 && id < MAX_RBUF && rbufs[id].live) ? rbufs[id].addr : 0;
}

int xmx_begin(void)
{
	if (!g.rready) FAIL("resident runtime not initialised", 0);
	vkResetCommandBuffer(g.rcb, 0);
	VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	VkResult r = vkBeginCommandBuffer(g.rcb, &bi);
	if (r) FAIL("begin resident recording", r);
	g.recording = 1;
	g.recorded = 0;
	return 0;
}

/* One global barrier between passes. Over-synchronised — consecutive independent
 * dispatches could overlap — but every pass here consumes the previous one's output,
 * so tracking finer dependencies would buy nothing. */
static void barrier(void)
{
	VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			       .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			       .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
	vkCmdPipelineBarrier(g.rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

int xmx_rec_gemm(int a, int b, int c, unsigned M, unsigned N, unsigned K, unsigned batch,
		 unsigned sa, unsigned sb, unsigned sc, unsigned bt,
		 unsigned lda, unsigned ldb, unsigned ldc,
		 unsigned oa, unsigned ob, unsigned oc)
{
	if (!g.recording) FAIL("not recording", 0);
	struct push p = { .a = addr_of(a), .b = addr_of(b), .c = addr_of(c),
			  .m = M, .n = N, .k = K, .batch = batch,
			  .sa = sa, .sb = sb, .sc = sc, .flags = bt,
			  .lda = lda, .ldb = ldb, .ldc = ldc };
	if (!p.a || !p.b || !p.c) FAIL("gemm operand is not a live buffer", 0);
	/* Element offsets are folded into the addresses, so a sub-matrix needs no shader
	 * support: A and B are half, and C is float unless the epilogue narrows it. */
	p.a += (uint64_t)oa * 2; p.b += (uint64_t)ob * 2;
	p.c += (uint64_t)oc * ((bt & 0x1000u) ? 2 : 4);
	/* The register-tiled kernel keeps a 16x32 block of the output in one subgroup's
	 * registers, which needs both extents to be a whole block; the 8x16 kernel takes
	 * everything else. Slice writes make the N test exact rather than conservative —
	 * a tile past the slice would land in the neighbouring one.
	 *
	 * 16x32 and not larger: a 32x32 block halves the workgroup count again and is
	 * *slower* in a frame, because the shapes here then stop having enough workgroups
	 * to fill the machine. It still wins in isolation, which is why the two disagree.
	 * The environment overrides exist so that trade can be re-measured. */
	int tiled = M % g.tilem == 0 && N % g.tilen == 0 && K >= g.tiling;
	vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, tiled ? g.rtiled : g.rgemm);
	vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
	if (tiled) vkCmdDispatch(g.rcb, N / g.tilen, M / g.tilem, batch ? batch : 1);
	else       vkCmdDispatch(g.rcb, (N + 15) / 16, (M + 7) / 8, batch ? batch : 1);
	barrier();
	g.recorded++;
	return 0;
}

int xmx_rec_unary(unsigned kind, int a, int b, int c, int d, unsigned n, unsigned channels,
		  float p0, unsigned batch, unsigned sa, unsigned sb, unsigned sc, unsigned k)
{
	if (!g.recording) FAIL("not recording", 0);
	struct push p = { .a = addr_of(a), .b = addr_of(b), .c = addr_of(c), .d = addr_of(d),
			  .m = n, .n = channels, .flags = kind, .p0 = p0,
			  .batch = batch, .sa = sa, .sb = sb, .sc = sc, .k = k };
	if (!p.a || !p.c) FAIL("unary operand is not a live buffer", 0);
	vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, g.runary);
	vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
	vkCmdDispatch(g.rcb, (n + 255) / 256, 1, 1);
	barrier();
	g.recorded++;
	return 0;
}

/* Row-wise passes: the cosine publish reduces 32 channels through the kernel's own
 * fragment tree, the softmax reduces a window's tokens. One invocation per row. */
int xmx_rec_row(unsigned kind, int a, int b, int c, int d, unsigned rows, unsigned width,
		unsigned heads, unsigned scaled, unsigned stride, float cap)
{
	if (!g.recording) FAIL("not recording", 0);
	struct push p = { .a = addr_of(a), .b = addr_of(b), .c = addr_of(c), .d = addr_of(d),
			  .m = rows, .n = width, .k = scaled, .batch = heads, .flags = kind,
			  .sa = stride, .p0 = cap };
	if (!p.a || !p.c) FAIL("row operand is not a live buffer", 0);
	vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, g.rrow);
	vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
	vkCmdDispatch(g.rcb, (rows + 31) / 32, 1, 1);
	barrier();
	g.recorded++;
	return 0;
}

/* The temporal path's five-tap reprojection: one invocation per pixel. */
int xmx_rec_history(int history, int motion, int out, unsigned pixels, unsigned channels,
		    unsigned height, unsigned width, unsigned absolute)
{
	if (!g.recording) FAIL("not recording", 0);
	struct push p = { .a = addr_of(history), .b = addr_of(motion), .c = addr_of(out),
			  .m = pixels, .n = channels, .k = height, .batch = width,
			  .flags = absolute };
	if (!p.a || !p.b || !p.c) FAIL("history operand is not a live buffer", 0);
	vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, g.rhistory);
	vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
	vkCmdDispatch(g.rcb, (pixels + 63) / 64, 1, 1);
	barrier();
	g.recorded++;
	return 0;
}

int xmx_submit(void)
{
	if (!g.recording) FAIL("not recording", 0);
	g.recording = 0;
	VkResult r = vkEndCommandBuffer(g.rcb);
	if (r) FAIL("end resident recording", r);
	vkResetFences(g.dev, 1, &g.rfence);
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
			    .pCommandBuffers = &g.rcb };
	if ((r = vkQueueSubmit(g.q, 1, &si, g.rfence))) FAIL("resident submit", r);
	if ((r = vkWaitForFences(g.dev, 1, &g.rfence, VK_TRUE, 60ull * 1000000000ull)))
		FAIL("resident fence wait", r);
	return g.recorded;
}
