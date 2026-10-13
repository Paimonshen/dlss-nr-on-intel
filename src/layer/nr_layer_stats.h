/* Optional host-side timings. GPU fence waits include scheduling and the game's
 * present waits; they are not GPU timestamp measurements of the copy itself.
 * Only sampled requests call the clock. Keep a bounded rolling window per
 * swapchain/extent/mode so resizing cannot mix incompatible frame costs. */
#ifndef NR_LAYER_STATS_H
#define NR_LAYER_STATS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define NR_TIMING_WINDOW 128u
#define NR_TIMING_SUMMARY_EVERY 8u
enum nr_timing_metric {
    NR_TIME_CAPTURE, NR_TIME_READBACK_WAIT, NR_TIME_EXCHANGE,
    NR_TIME_RETURN, NR_TIME_WRITEBACK_WAIT, NR_TIME_REQUEST,
    NR_TIME_PREVIOUS_RESPONSE, NR_TIME_METRICS
};
static const char *const nr_timing_names[NR_TIME_METRICS] = {
    "capture", "readback_wait", "exchange", "return", "writeback_wait",
    "request", "previous_response"
};
struct nr_timing_sample {
    uint64_t us[NR_TIME_METRICS];
    unsigned valid;
};
struct nr_layer_timing {
    int configured, enabled, active, mode, transport;
    unsigned every, width, height, family, count, cursor, since_summary, group;
    uint64_t session, chain, device, requests, delivered, failed, first_frame, last_frame;
    uint64_t started;
    struct nr_timing_sample pending, samples[NR_TIMING_WINDOW];
    const char *path;
};

static uint64_t nr_time_us(void)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart * 1000000.0 / frequency.QuadPart);
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u;
#endif
}

static int nr_time_compare(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Nearest-rank percentile; ignore absent stages (e.g. first async reply). */
static uint64_t nr_timing_percentile(const struct nr_layer_timing *stats,
                                    unsigned metric, unsigned percentile,
                                    unsigned *observations)
{
    uint64_t sorted[NR_TIMING_WINDOW];
    unsigned count = 0;
    for (unsigned i = 0; i < stats->count; i++)
        if (stats->samples[i].valid & (1u << metric))
            sorted[count++] = stats->samples[i].us[metric];
    if (observations) *observations = count;
    if (!count) return 0;
    qsort(sorted, count, sizeof sorted[0], nr_time_compare);
    unsigned rank = (count * percentile + 99u) / 100u;
    return sorted[rank ? rank - 1u : 0u];
}

static void nr_timing_write(FILE *out, const struct nr_layer_timing *stats)
{
    fprintf(out, "{\"kind\":\"nr_layer_timing\",\"session\":%llu,\"device\":%llu,"
        "\"swapchain\":%llu,\"width\":%u,\"height\":%u,\"family\":%u,"
        "\"mode\":\"%s\",\"transport\":\"%s\",\"sample_every\":%u,"
        "\"requests\":%llu,\"delivered\":%llu,\"undelivered\":%llu,"
        "\"first_frame\":%llu,\"last_frame\":%llu,\"samples\":%u,\"stages_ms\":{",
        (unsigned long long)stats->session, (unsigned long long)stats->device,
        (unsigned long long)stats->chain, stats->width, stats->height, stats->family,
        stats->mode ? "async" : "sync", stats->transport ? "gpu" : "cpu",
        stats->every, (unsigned long long)stats->requests,
        (unsigned long long)stats->delivered, (unsigned long long)stats->failed,
        (unsigned long long)stats->first_frame, (unsigned long long)stats->last_frame,
        stats->count);
    for (unsigned metric = 0; metric < NR_TIME_METRICS; metric++) {
        unsigned n;
        uint64_t p50 = nr_timing_percentile(stats, metric, 50, &n);
        uint64_t p95 = nr_timing_percentile(stats, metric, 95, NULL);
        fprintf(out, "%s\"%s\":{\"n\":%u,\"p50\":%.3f,\"p95\":%.3f}",
            metric ? "," : "", nr_timing_names[metric], n,
            (double)p50 / 1000.0, (double)p95 / 1000.0);
    }
    fputs("}}\n", out);
}

static void nr_timing_summary(struct nr_layer_timing *stats)
{
    if (!stats->enabled || !stats->count || !stats->since_summary) return;
    fputs("[nr_layer] timing: ", stderr);
    nr_timing_write(stderr, stats);
    /* Opt-in append-only diagnostic file; never put file I/O on every present. */
    if (stats->path && *stats->path) {
        FILE *out = fopen(stats->path, "a");
        if (out) { nr_timing_write(out, stats); fclose(out); }
    }
    stats->since_summary = 0;
}

static void nr_timing_configure(struct nr_layer_timing *stats)
{
    if (stats->configured) return;
    stats->configured = 1;
    const char *on = getenv("NR_LAYER_TIMING");
    stats->enabled = on && strcmp(on, "0") != 0;
    stats->every = 30;
    const char *every = getenv("NR_LAYER_TIMING_EVERY");
    if (every) {
        char *end;
        unsigned long value = strtoul(every, &end, 10);
        if (end != every && !*end && value >= 1 && value <= 1000000)
            stats->every = (unsigned)value;
    }
    stats->path = getenv("NR_LAYER_TIMING_FILE");
    if (stats->enabled) stats->session = nr_time_us();
}

static void nr_timing_begin(struct nr_layer_timing *stats, uint64_t device,
                            uint64_t chain, unsigned width, unsigned height,
                            unsigned family, uint64_t frame, int mode, int transport)
{
    nr_timing_configure(stats);
    if (!stats->enabled) return;
    if (stats->chain != chain || stats->width != width || stats->height != height ||
        stats->family != family || stats->mode != mode || stats->transport != transport) {
        nr_timing_summary(stats);
        stats->count = stats->cursor = stats->since_summary = 0;
        stats->group++;
        stats->requests = stats->delivered = stats->failed = 0;
        stats->first_frame = frame;
        stats->chain = chain; stats->width = width; stats->height = height;
        stats->family = family; stats->mode = mode; stats->transport = transport;
    }
    stats->device = device;
    stats->last_frame = frame;
    stats->requests++;
    stats->active = (stats->requests - 1u) % stats->every == 0;
    if (stats->active) {
        memset(&stats->pending, 0, sizeof stats->pending);
        stats->started = nr_time_us();
    }
}

static void nr_timing_add(struct nr_layer_timing *stats, unsigned metric, uint64_t us)
{
    if (!stats->active) return;
    stats->pending.us[metric] += us;
    stats->pending.valid |= 1u << metric;
}

static void nr_timing_finish(struct nr_layer_timing *stats, int delivered)
{
    if (!stats->enabled) return;
    if (delivered) stats->delivered++; else stats->failed++;
    if (!stats->active) return;
    nr_timing_add(stats, NR_TIME_REQUEST, nr_time_us() - stats->started);
    stats->samples[stats->cursor] = stats->pending;
    stats->cursor = (stats->cursor + 1u) % NR_TIMING_WINDOW;
    if (stats->count < NR_TIMING_WINDOW) stats->count++;
    stats->active = 0;
    stats->since_summary++;
    if (stats->since_summary >= NR_TIMING_SUMMARY_EVERY) nr_timing_summary(stats);
}

#endif
