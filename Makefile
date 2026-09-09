# Everything the resident path needs: one shared library and five compute shaders.
# The shaders are plain GLSL compiled to SPIR-V; libxmx keeps the Vulkan context alive
# across calls so a block records as one command buffer.

GLSL    := glslangValidator --target-env vulkan1.3
CFLAGS  := -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -Iwork/vulkan-headers/include
SHADERS := work/gemm_resident.spv work/resident.spv work/attention.spv \
           work/history.spv work/gemm_coopmat.spv work/gemm_batched.spv \
           work/gemm_f16acc.spv

all: work/libxmx.so $(SHADERS)

work/libxmx.so: src/gpu/libxmx.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lvulkan

work/gemm_resident.spv: src/gpu/gemm_resident.comp
	$(GLSL) -o $@ $<
work/resident.spv: src/gpu/resident.comp
	$(GLSL) -o $@ $<
work/attention.spv: src/gpu/attention.comp
	$(GLSL) -o $@ $<
work/history.spv: src/gpu/history.comp
	$(GLSL) -o $@ $<
work/gemm_coopmat.spv: src/gpu/gemm_coopmat.comp
	$(GLSL) -o $@ $<
work/gemm_batched.spv: src/gpu/gemm_coopmat_batched.comp
	$(GLSL) -o $@ $<
work/gemm_f16acc.spv: src/gpu/gemm_coopmat_f16acc.comp
	$(GLSL) -o $@ $<

test: all
	python3 src/gpu/test_epilogue.py
	python3 src/gpu/test_resident.py
	python3 src/ref/test_nr_model.py

.PHONY: all test
