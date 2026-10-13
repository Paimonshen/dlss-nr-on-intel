/* Actual local Win32 pipe smoke. No Vulkan loader, GPU or daemon involved. */
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
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "nr_gpu_rpc.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s (Win32 %lu)\n", __LINE__, #x, (unsigned long)GetLastError()); ExitProcess(1); } } while (0)
struct fixture {
    HANDLE pipe;
    struct nr_gpu_packet expected;
    int bad_session;
    int passed;
};
static DWORD WINAPI serve(void *argument)
{
    struct fixture *f = argument;
    unsigned char request[NR_GPU_RPC_REQUEST_BYTES];
    if (!ConnectNamedPipe(f->pipe, NULL) && GetLastError() != ERROR_PIPE_CONNECTED) return 1;
    DWORD read_bytes = 0;
    while (read_bytes < sizeof request) {
        DWORD got = 0;
        if (!ReadFile(f->pipe, request + read_bytes, (DWORD)sizeof request - read_bytes, &got, NULL) || !got) return 2;
        read_bytes += got;
    }
    struct nr_gpu_header h;
    struct nr_gpu_packet packet;
    memcpy(&h, request, sizeof h);
    memcpy(&packet, request + sizeof h, sizeof packet);
    if (h.magic != NR_GPU_MAGIC || h.width != packet.width || h.height != packet.height
        || h.format != packet.format || memcmp(&packet, &f->expected, sizeof packet)) return 3;
    uint64_t session = packet.session + (uint64_t)f->bad_session;
    struct nr_gpu_reply reply = {NR_GPU_SUCCESS, GetCurrentProcessId(), (uint32_t)session, (uint32_t)(session >> 32)};
    /* Give the client actual pending reads and repeated short completions. */
    for (DWORD i = 0; i < sizeof reply; ++i) {
        DWORD sent = 0;
        Sleep(1);
        if (!WriteFile(f->pipe, (unsigned char *)&reply + i, 1, &sent, NULL) || sent != 1) return 4;
    }
    if (!FlushFileBuffers(f->pipe)) return 5;
    f->passed = 1;
    return 0;
}
int main(void)
{
    for (unsigned run = 0; run < 4; ++run) {
        char path[128];
        snprintf(path, sizeof path, "\\\\.\\pipe\\nr-gpu-rpc-smoke-%lu-%u", (unsigned long)GetCurrentProcessId(), run);
        struct fixture f = {0};
        f.pipe = CreateNamedPipeA(path, PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                 1, 4096, 4096, 0, NULL);
        CHECK(f.pipe != INVALID_HANDLE_VALUE);
        f.expected.action = run == 3 ? NR_GPU_HELLO : run;
        f.expected.pid = GetCurrentProcessId(); f.expected.version = NR_GPU_VERSION;
        f.expected.width = 1280; f.expected.height = 720; f.expected.format = VK_FORMAT_B8G8R8A8_UNORM;
        f.expected.input_memory_type = 1; f.expected.output_memory_type = 2;
        f.expected.session = UINT64_C(0x9876543210203040) + run;
        f.bad_session = run == 3;
        HANDLE thread = CreateThread(NULL, 0, serve, &f, 0, NULL);
        CHECK(thread);
        struct nr_gpu_reply reply;
        DWORD peer = 0;
        /* Exercise the 3-argument wrapper and the optional OS peer result. */
        int result = run ? nr_gpu_rpc_peer(path, &f.expected, &reply, &peer)
                         : nr_gpu_rpc(path, &f.expected, &reply);
        DWORD error = GetLastError();
        if (run) CHECK(peer == GetCurrentProcessId());
        CHECK(result == !f.bad_session);
        if (result) CHECK(reply.status == NR_GPU_SUCCESS && reply.detail == GetCurrentProcessId());
        else CHECK(error == ERROR_INVALID_DATA && !reply.detail);
        CHECK(WaitForSingleObject(thread, 3000) == WAIT_OBJECT_0 && f.passed);
        CHECK(CloseHandle(thread)); CHECK(CloseHandle(f.pipe));
    }
    nr_gpu_rpc_reap(); CHECK(!nr_gpu_rpc_retired);
    puts("GPU control RPC Win32: real HELLO/FRAME/RELEASE pipes, short pending reads, peer PID and mismatched-session rejection OK");
    return 0;
}
