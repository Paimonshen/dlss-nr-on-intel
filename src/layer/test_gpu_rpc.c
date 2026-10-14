/* Win32 control transport contract, all I/O mocked. No GPU or daemon required. */
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

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define H(v) ((HANDLE)(uintptr_t)(v))
static ULONGLONG clock_ms;
static DWORD mock_error;
static unsigned opens, closes, events, writes, reads, cancels, drains;
static unsigned write_used, read_used;
static unsigned char request_bytes[sizeof(struct nr_gpu_header) + sizeof(struct nr_gpu_packet)];
static struct nr_gpu_reply server_reply;
static int busy_forever, missing_forever, denied, state_error, event_error;
static int server_pid_error;
static int pending_read, timeout_read, delayed_cancel, delayed_complete;
static int pending_write, timeout_write, broken_write, zero_write, active_writing;
static int cancel_not_found, wait_failed, broken_read, zero_read;
static DWORD write_delay, last_wait;
static OVERLAPPED *active_overlapped;
static void *active_buffer;
static DWORD active_length;
static int cancelled;

static DWORD WINAPI m_GetLastError(void) { return mock_error; }
static void WINAPI m_SetLastError(DWORD error) { mock_error = error; }
static ULONGLONG WINAPI m_GetTickCount64(void) { return clock_ms; }
static void WINAPI m_Sleep(DWORD delay) { clock_ms += delay; }
static HANDLE WINAPI m_CreateFileA(LPCSTR path, DWORD access, DWORD sharing,
    LPSECURITY_ATTRIBUTES security, DWORD disposition, DWORD flags, HANDLE template_file)
{
    CHECK(!strcmp(path, "\\\\.\\pipe\\nr-rpc-test"));
    CHECK(access == (GENERIC_READ | GENERIC_WRITE) && !sharing && !security);
    CHECK(disposition == OPEN_EXISTING && flags == FILE_FLAG_OVERLAPPED && !template_file);
    ++opens;
    if (busy_forever || missing_forever || denied) {
        mock_error = denied ? ERROR_ACCESS_DENIED : busy_forever ? ERROR_PIPE_BUSY : ERROR_FILE_NOT_FOUND;
        return INVALID_HANDLE_VALUE;
    }
    return H(10);
}
static BOOL WINAPI m_SetNamedPipeHandleState(HANDLE pipe, LPDWORD mode, LPDWORD max_count, LPDWORD collect)
{
    CHECK(pipe == H(10) && *mode == PIPE_READMODE_BYTE && !max_count && !collect);
    if (state_error) { mock_error = ERROR_ACCESS_DENIED; return FALSE; }
    return TRUE;
}
static BOOL WINAPI m_GetNamedPipeServerProcessId(HANDLE pipe, PULONG pid)
{
    CHECK(pipe == H(10));
    if (server_pid_error) { mock_error = ERROR_ACCESS_DENIED; return FALSE; }
    *pid = 321; return TRUE;
}
static HANDLE WINAPI m_CreateEventW(LPSECURITY_ATTRIBUTES security, BOOL manual, BOOL initial, LPCWSTR name)
{
    CHECK(!security && manual && !initial && !name);
    if (event_error) { mock_error = ERROR_NOT_ENOUGH_MEMORY; return NULL; }
    ++events; return H(20);
}
static BOOL WINAPI m_ResetEvent(HANDLE event) { CHECK(event == H(20) && !active_overlapped); return TRUE; }
static BOOL WINAPI m_CloseHandle(HANDLE handle)
{
    CHECK((handle == H(10) || handle == H(20)) && !active_overlapped);
    ++closes; mock_error = ERROR_INVALID_HANDLE; return TRUE;
}
static void write_chunk(const void *buffer, DWORD length, DWORD *transferred)
{
    DWORD chunk = length < 7u ? length : 7u;
    CHECK(write_used + chunk <= sizeof request_bytes);
    memcpy(request_bytes + write_used, buffer, chunk);
    write_used += chunk; *transferred = chunk;
}
static BOOL WINAPI m_WriteFile(HANDLE pipe, LPCVOID buffer, DWORD length, LPDWORD transferred, LPOVERLAPPED overlapped)
{
    CHECK(pipe == H(10) && overlapped && overlapped->hEvent == H(20));
    ++writes; clock_ms += write_delay;
    if (broken_write) { mock_error = ERROR_BROKEN_PIPE; return FALSE; }
    if (zero_write) { *transferred = 0; return TRUE; }
    if (pending_write) {
        CHECK(!active_overlapped);
        active_overlapped = overlapped; active_buffer = (void *)buffer; active_length = length;
        active_writing = 1; mock_error = ERROR_IO_PENDING; return FALSE;
    }
    write_chunk(buffer, length, transferred);
    return TRUE;
}
static void read_chunk(void *buffer, DWORD length, DWORD *transferred)
{
    DWORD remaining = (DWORD)sizeof server_reply - read_used;
    DWORD chunk = length < 3u ? length : 3u;
    CHECK(chunk <= remaining);
    memcpy(buffer, (unsigned char *)&server_reply + read_used, chunk);
    *transferred = chunk; read_used += chunk;
}
static BOOL WINAPI m_ReadFile(HANDLE pipe, LPVOID buffer, DWORD length, LPDWORD transferred, LPOVERLAPPED overlapped)
{
    CHECK(pipe == H(10) && overlapped && overlapped->hEvent == H(20));
    ++reads;
    if (broken_read) { mock_error = ERROR_BROKEN_PIPE; return FALSE; }
    if (zero_read) { *transferred = 0; return TRUE; }
    if (pending_read) {
        CHECK(!active_overlapped);
        active_overlapped = overlapped; active_buffer = buffer; active_length = length;
        active_writing = 0;
        mock_error = ERROR_IO_PENDING; return FALSE;
    }
    read_chunk(buffer, length, transferred); return TRUE;
}
static DWORD WINAPI m_WaitForSingleObject(HANDLE event, DWORD timeout)
{
    CHECK(event == H(20) && active_overlapped);
    if (cancelled) {
        CHECK(timeout == 100u);
        if (delayed_cancel && !delayed_complete) { clock_ms += timeout; return WAIT_TIMEOUT; }
        return WAIT_OBJECT_0;
    }
    last_wait = timeout;
    if (wait_failed) { mock_error = ERROR_INVALID_HANDLE; return WAIT_FAILED; }
    if ((active_writing && timeout_write) || (!active_writing && timeout_read)) {
        clock_ms += timeout; return WAIT_TIMEOUT;
    }
    ++clock_ms; return WAIT_OBJECT_0;
}
static BOOL WINAPI m_GetOverlappedResult(HANDLE pipe, LPOVERLAPPED overlapped, LPDWORD transferred, BOOL wait)
{
    CHECK(pipe == H(10) && overlapped == active_overlapped && !wait);
    ++drains;
    if (cancelled && delayed_cancel && !delayed_complete) { mock_error = ERROR_IO_INCOMPLETE; return FALSE; }
    active_overlapped = NULL;
    if (cancelled && !cancel_not_found) { mock_error = ERROR_OPERATION_ABORTED; return FALSE; }
    if (active_writing) write_chunk(active_buffer, active_length, transferred);
    else read_chunk(active_buffer, active_length, transferred);
    return TRUE;
}
static BOOL WINAPI m_CancelIoEx(HANDLE pipe, LPOVERLAPPED overlapped)
{
    CHECK(pipe == H(10) && overlapped == active_overlapped);
    ++cancels; cancelled = 1;
    if (cancel_not_found) { mock_error = ERROR_NOT_FOUND; return FALSE; }
    return TRUE;
}

#define GetLastError m_GetLastError
#define SetLastError m_SetLastError
#define GetTickCount64 m_GetTickCount64
#define Sleep m_Sleep
#define CreateFileA m_CreateFileA
#define SetNamedPipeHandleState m_SetNamedPipeHandleState
#define GetNamedPipeServerProcessId m_GetNamedPipeServerProcessId
#define CreateEventW m_CreateEventW
#define ResetEvent m_ResetEvent
#define CloseHandle m_CloseHandle
#define ReadFile m_ReadFile
#define WriteFile m_WriteFile
#define WaitForSingleObject m_WaitForSingleObject
#define GetOverlappedResult m_GetOverlappedResult
#define CancelIoEx m_CancelIoEx
#include "nr_gpu_rpc.h"

static struct nr_gpu_packet packet(void)
{
    struct nr_gpu_packet p = {0};
    p.action = NR_GPU_HELLO; p.width = 1280; p.height = 720; p.format = VK_FORMAT_B8G8R8A8_UNORM;
    p.version = NR_GPU_VERSION; p.session = UINT64_C(0x1020304050607080);
    return p;
}
static void reset(void)
{
    CHECK(!nr_gpu_rpc_retired && !active_overlapped);
    clock_ms = 50; mock_error = 0;
    opens = closes = events = writes = reads = cancels = drains = write_used = read_used = 0;
    busy_forever = missing_forever = denied = state_error = event_error = server_pid_error = 0;
    pending_read = timeout_read = delayed_cancel = delayed_complete = 0;
    pending_write = timeout_write = broken_write = zero_write = active_writing = 0;
    cancel_not_found = wait_failed = broken_read = zero_read = cancelled = 0;
    write_delay = last_wait = 0;
    memset(request_bytes, 0, sizeof request_bytes);
    struct nr_gpu_packet p = packet();
    server_reply.status = NR_GPU_SUCCESS; server_reply.detail = 321;
    server_reply.session_low = (uint32_t)p.session;
    server_reply.session_high = (uint32_t)(p.session >> 32);
}
static int exchange(struct nr_gpu_reply *reply)
{
    struct nr_gpu_packet p = packet();
    return nr_gpu_rpc("\\\\.\\pipe\\nr-rpc-test", &p, reply);
}
static void success_tests(void)
{
    struct nr_gpu_reply reply;
    reset(); CHECK(exchange(&reply));
    CHECK(reply.detail == 321 && opens == 1 && events == 1 && closes == 2 && !cancels);
    CHECK(writes > 1 && reads > 1 && write_used == sizeof request_bytes && read_used == sizeof reply);
    struct nr_gpu_header h; struct nr_gpu_packet p, expected = packet();
    memcpy(&h, request_bytes, sizeof h); memcpy(&p, request_bytes + sizeof h, sizeof p);
    CHECK(h.magic == NR_GPU_MAGIC && h.width == p.width && h.height == p.height && h.format == p.format);
    CHECK(!memcmp(&p, &expected, sizeof p));
    reset(); pending_read = 1; write_delay = 17;
    CHECK(exchange(&reply));
    CHECK(drains == reads && !cancels && closes == 2 && last_wait < NR_GPU_RPC_TIMEOUT_MS);
    reset(); pending_read = pending_write = 1;
    CHECK(exchange(&reply));
    CHECK(drains == reads + writes && closes == 2 && !cancels);
    reset(); server_reply.status = NR_GPU_UNSUPPORTED;
    CHECK(exchange(&reply) && reply.status == NR_GPU_UNSUPPORTED);
    reset(); struct nr_gpu_packet p2 = packet();
    p2.version = NR_GPU_FENCED_VERSION;
    CHECK(nr_gpu_rpc("\\\\.\\pipe\\nr-rpc-test", &p2, &reply));
}
static void rejection_tests(void)
{
    struct nr_gpu_reply reply;
    reset(); server_reply.session_high ^= 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_INVALID_DATA && closes == 2 && !reply.detail);
    reset(); server_reply.status = 3;
    CHECK(!exchange(&reply) && mock_error == ERROR_INVALID_DATA);
    reset(); server_reply.detail = 999;
    CHECK(!exchange(&reply) && mock_error == ERROR_INVALID_DATA && closes == 2);
    reset(); broken_read = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_BROKEN_PIPE && !cancels && closes == 2);
    reset(); zero_read = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_BAD_LENGTH && !reply.status);
    reset(); broken_write = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_BROKEN_PIPE && !reads && closes == 2);
    reset(); zero_write = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_BAD_LENGTH && !reads && closes == 2);
    reset(); denied = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_ACCESS_DENIED && opens == 1 && !events && !closes);
    reset(); state_error = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_ACCESS_DENIED && closes == 1 && !events);
    reset(); event_error = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_NOT_ENOUGH_MEMORY && closes == 1);
    reset(); server_pid_error = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_ACCESS_DENIED && closes == 1 && !events);
    reset(); struct nr_gpu_packet p = packet();
    CHECK(!nr_gpu_rpc("C:\\not-a-pipe", &p, &reply) && mock_error == ERROR_INVALID_PARAMETER && !opens);
    p.version = 0;
    CHECK(!nr_gpu_rpc("\\\\.\\pipe\\nr-rpc-test", &p, &reply) && !opens);
    p = packet(); p.session = 0;
    CHECK(!nr_gpu_rpc("\\\\.\\pipe\\nr-rpc-test", &p, &reply) && !opens);
}
static void timeout_tests(void)
{
    struct nr_gpu_reply reply;
    reset(); busy_forever = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT && clock_ms == 10050 && !events && !closes);
    reset(); missing_forever = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT && clock_ms == 10050);
    reset(); pending_read = timeout_read = 1; write_delay = 17;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT);
    CHECK(cancels == 1 && drains == 1 && closes == 2 && clock_ms == 10050);
    CHECK(last_wait == NR_GPU_RPC_TIMEOUT_MS - writes * write_delay);
    reset(); pending_write = timeout_write = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT && cancels == 1 && drains == 1 && closes == 2);
    CHECK(writes == 1 && !reads && !write_used && clock_ms == 10050);
    reset(); pending_read = timeout_read = cancel_not_found = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT && cancels == 1 && closes == 2);
    reset(); pending_read = wait_failed = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_INVALID_HANDLE && cancels == 1 && closes == 2);
    reset(); pending_read = timeout_read = delayed_cancel = 1;
    CHECK(!exchange(&reply) && mock_error == ERROR_TIMEOUT && cancels == 1 && !closes);
    CHECK(nr_gpu_rpc_retired && nr_gpu_rpc_retired->pending && active_overlapped);
    CHECK(clock_ms == 10150 && !reply.detail);
    nr_gpu_rpc_reap(); CHECK(nr_gpu_rpc_retired && !closes && mock_error == ERROR_TIMEOUT);
    delayed_complete = 1; nr_gpu_rpc_reap();
    CHECK(!nr_gpu_rpc_retired && !active_overlapped && closes == 2 && mock_error == ERROR_TIMEOUT);
}
static void peer_tests(void)
{
    struct nr_gpu_reply reply;
    struct nr_gpu_packet p = packet();
    DWORD peer = 777;
    reset();
    CHECK(nr_gpu_rpc_peer("\\\\.\\pipe\\nr-rpc-test", &p, &reply, &peer) && peer == 321);
    reset(); pending_read = timeout_read = 1;
    CHECK(!nr_gpu_rpc_peer("\\\\.\\pipe\\nr-rpc-test", &p, &reply, &peer));
    CHECK(peer == 321 && !reply.detail && mock_error == ERROR_TIMEOUT);
    reset(); denied = 1;
    CHECK(!nr_gpu_rpc_peer("\\\\.\\pipe\\nr-rpc-test", &p, &reply, &peer) && !peer);
    reset(); server_reply.detail = 999;
    CHECK(!nr_gpu_rpc_peer("\\\\.\\pipe\\nr-rpc-test", &p, &reply, &peer));
    CHECK(peer == 321 && mock_error == ERROR_INVALID_DATA && !reply.detail);
}
int main(void)
{
    success_tests(); rejection_tests(); timeout_tests(); peer_tests();
    puts("GPU control RPC mock: partial IO, session echo, shared deadline, disconnect, cancellation race, delayed completion retention and cleanup OK");
    return 0;
}
