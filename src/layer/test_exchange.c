/* Exercise the game's actual socket exchange with SIGPIPE's default disposition. */
#include "nr_layer.c"
#include <signal.h>

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    signal(SIGPIPE, SIG_DFL);
    socket_path = argv[1];
    int reject = strcmp(argv[2], "reject") == 0;
    unsigned width = reject ? 2048 : 64, height = reject ? 1024 : 32;
    size_t bytes = (size_t)width * height * 4;
    unsigned char *payload = calloc(bytes, 1), *reply = malloc(bytes);
    if (!payload || !reply) return 2;
    uint32_t header[] = { 0x304E524E, width, height, 44 };
    int result = exchange(header, sizeof header, payload, bytes, reply);
    int ok = reject ? result == -1 : result == 0 && memcmp(payload, reply, bytes) == 0;
    free(payload); free(reply);
    return ok ? 0 : 1;
}
