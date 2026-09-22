# Build the resident Vulkan runtime, game layer and compute shaders.
# The shaders are plain GLSL compiled to SPIR-V; libxmx keeps the Vulkan context alive
# across calls so a block records as one command buffer.

GLSL    := glslangValidator --target-env vulkan1.3 -Isrc/gpu
CFLAGS  := -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -Iwork/vulkan-headers/include
SHADERS := work/gemm_resident.spv work/gemm_tiled.spv work/gemm_staged.spv \
           work/resident.spv work/attention.spv \
           work/history.spv work/gemm_coopmat.spv work/gemm_batched.spv \
           work/gemm_f16acc.spv work/gemm_coopmat_int8.spv

all: work/libxmx.so work/libnr_layer.so work/libnr_image.so work/gemm_runner $(SHADERS)

work:
	mkdir -p $@

$(SHADERS) work/libxmx.so work/libnr_layer.so work/half_probe.spv work/attention_ab.spv: | work
work/libnr_image.so work/test_exchange work/test_settled work/libnr_layer32.so work/test_layer_loader work/test_present work/test_layer_loader32: | work

work/libxmx.so: src/gpu/libxmx.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lvulkan

# The host passes in C. Built for this machine: `-march=native`, so rebuild it rather
# than copy it. The FP16 casts and the separate multiply and add are the contract —
# fused multiply-add or fast maths would change the last bit and the output must be
# byte-identical to the NumPy it replaces (src/ref/test_native_image.py).
work/libnr_image.so: src/ref/nr_image.c Makefile | work
	$(CC) -O3 -march=native -fPIC -Wall -Wextra -ffp-contract=off -fno-fast-math \
	      -shared -o $@ $<

# The Vulkan layer that puts the pass inside a running game.
work/libnr_layer.so: src/layer/nr_layer.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lvulkan

work/libnr_layer32.so: src/layer/nr_layer.c
	$(CC) $(CFLAGS) -m32 -shared -o $@ $< -lvulkan

work/test_layer_loader: src/layer/test_layer_loader.c
	$(CC) $(CFLAGS) -o $@ $< -lvulkan

work/test_present: src/layer/test_present.c
	$(CC) $(CFLAGS) -o $@ $< -lvulkan

work/test_layer_loader32: src/layer/test_layer_loader.c
	$(CC) $(CFLAGS) -m32 -o $@ $< -lvulkan

work/test_settled: src/layer/test_settled.c src/layer/nr_layer.c
	$(CC) $(CFLAGS) -Isrc/layer -o $@ $< -lvulkan

work/test_exchange: src/layer/test_exchange.c src/layer/nr_layer.c
	$(CC) $(CFLAGS) -o $@ $< -lvulkan -lpthread

work/gemm_resident.spv: src/gpu/gemm_resident.comp src/gpu/publish.glsl src/gpu/specialize.glsl
	$(GLSL) -o $@ $<
# the same source, with a 16x32 block of the output held in one subgroup's registers
work/gemm_tiled.spv: src/gpu/gemm_resident.comp src/gpu/publish.glsl src/gpu/specialize.glsl Makefile
	$(GLSL) -DRM=2 -DRN=2 -o $@ $<
work/gemm_staged.spv: src/gpu/gemm_staged.comp src/gpu/publish.glsl src/gpu/specialize.glsl
	$(GLSL) -o $@ $<
work/resident.spv: src/gpu/resident.comp src/gpu/publish.glsl src/gpu/specialize.glsl
	$(GLSL) -o $@ $<
work/attention.spv: src/gpu/attention.comp src/gpu/publish.glsl src/gpu/specialize.glsl
	$(GLSL) -o $@ $<
work/attention_ab.spv: src/gpu/attention.comp src/gpu/publish.glsl src/gpu/specialize.glsl
	$(GLSL) -DSOFTMAX_AB -o $@ $<
work/history.spv: src/gpu/history.comp
	$(GLSL) -o $@ $<
# Configuration 4, the integer twin. Built always; used only where
# notes/improve-int8-bottleneck.md measured the trade as worth taking.
work/gemm_coopmat_int8.spv: src/gpu/gemm_coopmat_int8.comp
	$(GLSL) -o $@ $<

# The one-shot runner behind test_gemm.py and test_gemm_int8.py: its own
# instance and device per call, and the operand width read from the files.
work/gemm_runner: src/gpu/gemm_runner.c | work
	$(CC) $(CFLAGS) -o $@ $< -lvulkan

work/gemm_coopmat.spv: src/gpu/gemm_coopmat.comp
	$(GLSL) -o $@ $<
work/gemm_batched.spv: src/gpu/gemm_coopmat_batched.comp
	$(GLSL) -o $@ $<
work/gemm_f16acc.spv: src/gpu/gemm_coopmat_f16acc.comp
	$(GLSL) -o $@ $<

work/half_probe.spv: src/bench/half_probe.comp
	$(GLSL) -o $@ $<

bench: all work/half_probe.spv
	python3 src/bench/half_probe.py
	python3 src/bench/split_cost.py

test: all work/attention_ab.spv work/test_exchange work/test_settled work/test_present
	python3 src/gpu/test_ffn_batch.py --gpu
	python3 src/gpu/test_gemm_contract.py
	python3 src/gpu/test_input_fp16.py
	python3 src/gpu/test_compact_head.py
	python3 src/gpu/test_joint_qkv.py
	python3 src/layer/test_daemon.py
	work/test_settled
	python3 src/layer/test_ui_mask.py
	python3 src/layer/test_present.py
	python3 src/layer/test_temporal.py
	python3 src/layer/test_toggle.py
	python3 src/layer/test_panel.py
	python3 src/tools/publish_check.py
	python3 src/tools/claims_check.py
	python3 src/gpu/test_gemm_int8.py
	python3 src/gpu/test_epilogue.py
	python3 src/gpu/test_specialization.py
	python3 src/gpu/test_softmax_pack.py
	python3 src/gpu/test_qkv_fusion.py
	python3 src/gpu/test_scratch_arena.py
	python3 src/gpu/test_graph.py
	python3 src/gpu/test_frame_execution.py
	python3 src/gpu/test_resident.py
	python3 src/ref/test_frame_cache.py
	python3 src/ref/test_native_image.py
	python3 src/ref/test_nr_model.py
	python3 src/ref/test_temporal_controls.py

# Focused checks for the FFN schedule, including a complete frame with both variants.
# Repeat with XMX_STAGING=1 to cover device buffers without host mappings.
test-ffn: all
	python3 src/gpu/test_ffn_batch.py --gpu
	python3 src/gpu/test_frame_execution.py

test-proton: all work/libnr_layer32.so work/test_layer_loader work/test_layer_loader32
	python3 src/layer/test_launcher.py
	python3 src/layer/prepare_layer.py work/layer-check
	VK_LAYER_PATH=$(CURDIR)/work/layer-check ENABLE_NR_LAYER=1 work/test_layer_loader
	VK_LAYER_PATH=$(CURDIR)/work/layer-check ENABLE_NR_LAYER=1 work/test_layer_loader32

# Everything the tracked tree must not contain, and — slower — everything the history
# must not either. `make test` runs the first; run the second before publishing.
publish-check:
	python3 src/tools/publish_check.py --history

.PHONY: all test test-ffn test-proton bench publish-check
