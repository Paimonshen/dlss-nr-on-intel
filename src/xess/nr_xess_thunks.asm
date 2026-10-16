; Written by gen_xess_exports.py from libxess.exports; not edited by hand.
; Each jumps through the pointer DllMain set from Intel's libxess.dll.
.data
PUBLIC nr_xess_forward
nr_xess_forward QWORD 68 DUP (0)
.code
nr_forward_INTC_D3D11_XessCreateContext PROC
	jmp QWORD PTR [nr_xess_forward + 0]
nr_forward_INTC_D3D11_XessCreateContext ENDP
nr_forward_INTC_D3D11_XessDestroyContext PROC
	jmp QWORD PTR [nr_xess_forward + 8]
nr_forward_INTC_D3D11_XessDestroyContext ENDP
nr_forward_INTC_D3D11_XessExecute PROC
	jmp QWORD PTR [nr_xess_forward + 16]
nr_forward_INTC_D3D11_XessExecute ENDP
nr_forward_INTC_D3D11_XessForceLegacyScaleFactors PROC
	jmp QWORD PTR [nr_xess_forward + 24]
nr_forward_INTC_D3D11_XessForceLegacyScaleFactors ENDP
nr_forward_INTC_D3D11_XessGetExposureMultiplier PROC
	jmp QWORD PTR [nr_xess_forward + 32]
nr_forward_INTC_D3D11_XessGetExposureMultiplier ENDP
nr_forward_INTC_D3D11_XessGetInitParams PROC
	jmp QWORD PTR [nr_xess_forward + 40]
nr_forward_INTC_D3D11_XessGetInitParams ENDP
nr_forward_INTC_D3D11_XessGetInputResolution PROC
	jmp QWORD PTR [nr_xess_forward + 48]
nr_forward_INTC_D3D11_XessGetInputResolution ENDP
nr_forward_INTC_D3D11_XessGetIntelXeFXVersion PROC
	jmp QWORD PTR [nr_xess_forward + 56]
nr_forward_INTC_D3D11_XessGetIntelXeFXVersion ENDP
nr_forward_INTC_D3D11_XessGetJitterScale PROC
	jmp QWORD PTR [nr_xess_forward + 64]
nr_forward_INTC_D3D11_XessGetJitterScale ENDP
nr_forward_INTC_D3D11_XessGetOptimalInputResolution PROC
	jmp QWORD PTR [nr_xess_forward + 72]
nr_forward_INTC_D3D11_XessGetOptimalInputResolution ENDP
nr_forward_INTC_D3D11_XessGetProfilingData PROC
	jmp QWORD PTR [nr_xess_forward + 80]
nr_forward_INTC_D3D11_XessGetProfilingData ENDP
nr_forward_INTC_D3D11_XessGetVelocityScale PROC
	jmp QWORD PTR [nr_xess_forward + 88]
nr_forward_INTC_D3D11_XessGetVelocityScale ENDP
nr_forward_INTC_D3D11_XessGetVersion PROC
	jmp QWORD PTR [nr_xess_forward + 96]
nr_forward_INTC_D3D11_XessGetVersion ENDP
nr_forward_INTC_D3D11_XessInit PROC
	jmp QWORD PTR [nr_xess_forward + 104]
nr_forward_INTC_D3D11_XessInit ENDP
nr_forward_INTC_D3D11_XessIsOptimalDriver PROC
	jmp QWORD PTR [nr_xess_forward + 112]
nr_forward_INTC_D3D11_XessIsOptimalDriver ENDP
nr_forward_INTC_D3D11_XessSelectNetworkModel PROC
	jmp QWORD PTR [nr_xess_forward + 120]
nr_forward_INTC_D3D11_XessSelectNetworkModel ENDP
nr_forward_INTC_D3D11_XessSetExposureMultiplier PROC
	jmp QWORD PTR [nr_xess_forward + 128]
nr_forward_INTC_D3D11_XessSetExposureMultiplier ENDP
nr_forward_INTC_D3D11_XessSetJitterScale PROC
	jmp QWORD PTR [nr_xess_forward + 136]
nr_forward_INTC_D3D11_XessSetJitterScale ENDP
nr_forward_INTC_D3D11_XessSetLoggingCallback PROC
	jmp QWORD PTR [nr_xess_forward + 144]
nr_forward_INTC_D3D11_XessSetLoggingCallback ENDP
nr_forward_INTC_D3D11_XessSetVelocityScale PROC
	jmp QWORD PTR [nr_xess_forward + 152]
nr_forward_INTC_D3D11_XessSetVelocityScale ENDP
nr_forward_INTC_D3D12_GetApplicationInfo PROC
	jmp QWORD PTR [nr_xess_forward + 160]
nr_forward_INTC_D3D12_GetApplicationInfo ENDP
nr_forward_INTC_D3D12_GetGpuCrashDumpBufferInfo PROC
	jmp QWORD PTR [nr_xess_forward + 168]
nr_forward_INTC_D3D12_GetGpuCrashDumpBufferInfo ENDP
nr_forward_INTC_D3D12_GpuCrashDumpOnCommandListCreate PROC
	jmp QWORD PTR [nr_xess_forward + 176]
nr_forward_INTC_D3D12_GpuCrashDumpOnCommandListCreate ENDP
nr_forward_INTC_D3D12_GpuCrashDumpOnCommandListDestroy PROC
	jmp QWORD PTR [nr_xess_forward + 184]
nr_forward_INTC_D3D12_GpuCrashDumpOnCommandListDestroy ENDP
nr_forward_INTC_D3D12_GpuCrashDumpOnDeviceDestroy PROC
	jmp QWORD PTR [nr_xess_forward + 192]
nr_forward_INTC_D3D12_GpuCrashDumpOnDeviceDestroy ENDP
nr_forward_INTC_D3D12_RegisterDriverCallbacks PROC
	jmp QWORD PTR [nr_xess_forward + 200]
nr_forward_INTC_D3D12_RegisterDriverCallbacks ENDP
nr_forward_xessAILGetDecision PROC
	jmp QWORD PTR [nr_xess_forward + 208]
nr_forward_xessAILGetDecision ENDP
nr_forward_xessAILGetVersion PROC
	jmp QWORD PTR [nr_xess_forward + 216]
nr_forward_xessAILGetVersion ENDP
nr_forward_xessD3D12BuildPipelines PROC
	jmp QWORD PTR [nr_xess_forward + 224]
nr_forward_xessD3D12BuildPipelines ENDP
nr_forward_xessD3D12CreateContext PROC
	jmp QWORD PTR [nr_xess_forward + 232]
nr_forward_xessD3D12CreateContext ENDP
nr_forward_xessD3D12GetInitParams PROC
	jmp QWORD PTR [nr_xess_forward + 240]
nr_forward_xessD3D12GetInitParams ENDP
nr_forward_xessD3D12GetProfilingData PROC
	jmp QWORD PTR [nr_xess_forward + 248]
nr_forward_xessD3D12GetProfilingData ENDP
nr_forward_xessD3D12GetResourcesToDump PROC
	jmp QWORD PTR [nr_xess_forward + 256]
nr_forward_xessD3D12GetResourcesToDump ENDP
nr_forward_xessDestroyContext PROC
	jmp QWORD PTR [nr_xess_forward + 264]
nr_forward_xessDestroyContext ENDP
nr_forward_xessForceLegacyScaleFactors PROC
	jmp QWORD PTR [nr_xess_forward + 272]
nr_forward_xessForceLegacyScaleFactors ENDP
nr_forward_xessGetContextParameterP PROC
	jmp QWORD PTR [nr_xess_forward + 280]
nr_forward_xessGetContextParameterP ENDP
nr_forward_xessGetExposureMultiplier PROC
	jmp QWORD PTR [nr_xess_forward + 288]
nr_forward_xessGetExposureMultiplier ENDP
nr_forward_xessGetInputResolution PROC
	jmp QWORD PTR [nr_xess_forward + 296]
nr_forward_xessGetInputResolution ENDP
nr_forward_xessGetIntelXeFXVersion PROC
	jmp QWORD PTR [nr_xess_forward + 304]
nr_forward_xessGetIntelXeFXVersion ENDP
nr_forward_xessGetJitterScale PROC
	jmp QWORD PTR [nr_xess_forward + 312]
nr_forward_xessGetJitterScale ENDP
nr_forward_xessGetMaxResponsiveMaskValue PROC
	jmp QWORD PTR [nr_xess_forward + 320]
nr_forward_xessGetMaxResponsiveMaskValue ENDP
nr_forward_xessGetOptimalInputResolution PROC
	jmp QWORD PTR [nr_xess_forward + 328]
nr_forward_xessGetOptimalInputResolution ENDP
nr_forward_xessGetPipelineBuildStatus PROC
	jmp QWORD PTR [nr_xess_forward + 336]
nr_forward_xessGetPipelineBuildStatus ENDP
nr_forward_xessGetProfilingData PROC
	jmp QWORD PTR [nr_xess_forward + 344]
nr_forward_xessGetProfilingData ENDP
nr_forward_xessGetProperties PROC
	jmp QWORD PTR [nr_xess_forward + 352]
nr_forward_xessGetProperties ENDP
nr_forward_xessGetVelocityScale PROC
	jmp QWORD PTR [nr_xess_forward + 360]
nr_forward_xessGetVelocityScale ENDP
nr_forward_xessGetVersion PROC
	jmp QWORD PTR [nr_xess_forward + 368]
nr_forward_xessGetVersion ENDP
nr_forward_xessIsOptimalDriver PROC
	jmp QWORD PTR [nr_xess_forward + 376]
nr_forward_xessIsOptimalDriver ENDP
nr_forward_xessSelectNetworkModel PROC
	jmp QWORD PTR [nr_xess_forward + 384]
nr_forward_xessSelectNetworkModel ENDP
nr_forward_xessSetContextParameterF PROC
	jmp QWORD PTR [nr_xess_forward + 392]
nr_forward_xessSetContextParameterF ENDP
nr_forward_xessSetExposureMultiplier PROC
	jmp QWORD PTR [nr_xess_forward + 400]
nr_forward_xessSetExposureMultiplier ENDP
nr_forward_xessSetLoggingCallback PROC
	jmp QWORD PTR [nr_xess_forward + 408]
nr_forward_xessSetLoggingCallback ENDP
nr_forward_xessSetMaxResponsiveMaskValue PROC
	jmp QWORD PTR [nr_xess_forward + 416]
nr_forward_xessSetMaxResponsiveMaskValue ENDP
nr_forward_xessStartDump PROC
	jmp QWORD PTR [nr_xess_forward + 424]
nr_forward_xessStartDump ENDP
nr_forward_xessTrinityCreateContext PROC
	jmp QWORD PTR [nr_xess_forward + 432]
nr_forward_xessTrinityCreateContext ENDP
nr_forward_xessTrinityGetRequiredDeviceExtensions PROC
	jmp QWORD PTR [nr_xess_forward + 440]
nr_forward_xessTrinityGetRequiredDeviceExtensions ENDP
nr_forward_xessTrinityGetRequiredDeviceFeatures PROC
	jmp QWORD PTR [nr_xess_forward + 448]
nr_forward_xessTrinityGetRequiredDeviceFeatures ENDP
nr_forward_xessTrinityGetRequiredInstanceExtensions PROC
	jmp QWORD PTR [nr_xess_forward + 456]
nr_forward_xessTrinityGetRequiredInstanceExtensions ENDP
nr_forward_xessVKAILGetDecision PROC
	jmp QWORD PTR [nr_xess_forward + 464]
nr_forward_xessVKAILGetDecision ENDP
nr_forward_xessVKBuildPipelines PROC
	jmp QWORD PTR [nr_xess_forward + 472]
nr_forward_xessVKBuildPipelines ENDP
nr_forward_xessVKCreateContext PROC
	jmp QWORD PTR [nr_xess_forward + 480]
nr_forward_xessVKCreateContext ENDP
nr_forward_xessVKExecute PROC
	jmp QWORD PTR [nr_xess_forward + 488]
nr_forward_xessVKExecute ENDP
nr_forward_xessVKGetInitParams PROC
	jmp QWORD PTR [nr_xess_forward + 496]
nr_forward_xessVKGetInitParams ENDP
nr_forward_xessVKGetRequiredDeviceExtensions PROC
	jmp QWORD PTR [nr_xess_forward + 504]
nr_forward_xessVKGetRequiredDeviceExtensions ENDP
nr_forward_xessVKGetRequiredDeviceFeatures PROC
	jmp QWORD PTR [nr_xess_forward + 512]
nr_forward_xessVKGetRequiredDeviceFeatures ENDP
nr_forward_xessVKGetRequiredInstanceExtensions PROC
	jmp QWORD PTR [nr_xess_forward + 520]
nr_forward_xessVKGetRequiredInstanceExtensions ENDP
nr_forward_xessVKGetResourcesToDump PROC
	jmp QWORD PTR [nr_xess_forward + 528]
nr_forward_xessVKGetResourcesToDump ENDP
nr_forward_xessVKInit PROC
	jmp QWORD PTR [nr_xess_forward + 536]
nr_forward_xessVKInit ENDP
END
