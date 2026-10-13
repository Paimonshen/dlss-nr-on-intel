/* Primary surface selection, independent of Vulkan dispatch and GPU execution.
 * The caller supplies only live swapchains belonging to one device, marks every
 * presented surface before selecting, and processes the selected handle only
 * when it occurs in the current present. Never substitute a service surface
 * merely because the main window is absent from this particular present. */
#ifndef NR_SURFACE_POLICY_H
#define NR_SURFACE_POLICY_H

#include <stddef.h>
#include <stdint.h>

#define NR_SURFACE_DEFAULT_TTL UINT64_C(120)

struct nr_surface_record {
    uint64_t handle, last_seen;
    uint32_t width, height;
    /* A zero tick is valid after counter wrap, so do not use last_seen==0 as
     * an unpresented sentinel. A newly created surface is initially unseen. */
    int seen;
};

struct nr_surface_selection {
    uint64_t handle;
    uint32_t width, height;
};

enum nr_surface_reason {
    NR_SURFACE_NONE,
    NR_SURFACE_LARGEST_RECENT,
    NR_SURFACE_CURRENT_RETAINED,
    NR_SURFACE_EXPLICIT
};

struct nr_surface_choice {
    struct nr_surface_selection selection;
    size_t index;                    /* SIZE_MAX means no eligible live surface. */
    uint64_t area, age;
    int changed, recent, preferred_unavailable;
    enum nr_surface_reason reason;
};

static inline void nr_surface_mark_presented(struct nr_surface_record *record, uint64_t now)
{
    record->last_seen = now;
    record->seen = 1;
}

static inline uint64_t nr_surface_area(const struct nr_surface_record *record)
{
    /* Each dimension is uint32, so its full product fits uint64. Cast before
     * multiplying; even 65536x65536 otherwise overflows a 32-bit product. */
    return (uint64_t)record->width * (uint64_t)record->height;
}

static inline int nr_surface_recent(const struct nr_surface_record *record,
                                     uint64_t now, uint64_t ttl)
{
    /* Unsigned modular subtraction deliberately supports a wrapping monotonic
     * presentation counter. Callers should configure a short TTL, not use a
     * wall clock or timestamps from different counters. */
    return record->seen && now - record->last_seen <= ttl;
}

static inline struct nr_surface_choice nr_surface_choose(
    const struct nr_surface_record *records, size_t count, uint64_t now, uint64_t ttl,
    struct nr_surface_selection current, uint64_t preferred_handle)
{
    struct nr_surface_choice choice = {0};
    choice.index = SIZE_MAX;
    choice.preferred_unavailable = preferred_handle != 0;
    if (!records) count = 0;

    /* An explicit user selection overrides freshness while the handle remains
     * live and usable. It does not cause capture of a surface absent from the
     * actual present. Destruction or a zero extent makes it unavailable. */
    if (preferred_handle)
        for (size_t i = 0; i < count; i++)
            if (records[i].handle == preferred_handle && records[i].width && records[i].height) {
                choice.index = i;
                choice.preferred_unavailable = 0;
                choice.reason = NR_SURFACE_EXPLICIT;
                break;
            }

    if (choice.index == SIZE_MAX)
        for (size_t i = 0; i < count; i++) {
            const struct nr_surface_record *candidate = &records[i];
            if (!candidate->handle || !candidate->width || !candidate->height ||
                !nr_surface_recent(candidate, now, ttl)) continue;
            uint64_t area = nr_surface_area(candidate);
            int choose = choice.index == SIZE_MAX || area > choice.area;
            if (!choose && area == choice.area) {
                uint64_t best_handle = records[choice.index].handle;
                /* Preserve the current choice on equal area. Otherwise use
                 * handle order so rearranging the caller's array cannot flicker. */
                choose = candidate->handle == current.handle ||
                    (best_handle != current.handle && candidate->handle < best_handle);
            }
            if (choose) {
                choice.index = i;
                choice.area = area;
            }
        }

    if (choice.index != SIZE_MAX) {
        const struct nr_surface_record *chosen = &records[choice.index];
        choice.selection.handle = chosen->handle;
        choice.selection.width = chosen->width;
        choice.selection.height = chosen->height;
        choice.area = nr_surface_area(chosen);
        choice.age = chosen->seen ? now - chosen->last_seen : UINT64_MAX;
        choice.recent = nr_surface_recent(chosen, now, ttl);
        if (choice.reason != NR_SURFACE_EXPLICIT)
            choice.reason = chosen->handle == current.handle
                ? NR_SURFACE_CURRENT_RETAINED : NR_SURFACE_LARGEST_RECENT;
    }
    choice.changed = choice.selection.handle != current.handle ||
        choice.selection.width != current.width || choice.selection.height != current.height;
    return choice;
}

#endif
