/* CPU-only surface policy cases: auxiliary windows must not switch network
 * dimensions/history while the main surface is still recently presented. */
#include "nr_surface_policy.h"
#include <assert.h>
#include <stdio.h>

static struct nr_surface_selection none;

static void test_main_and_service(void)
{
    struct nr_surface_record records[] = {
        { .handle = 10, .width = 1920, .height = 1080 },
        { .handle = 20, .width = 160, .height = 28 }
    };
    nr_surface_mark_presented(&records[0], 1);
    struct nr_surface_choice main = nr_surface_choose(records, 2, 1, 120, none, 0);
    assert(main.selection.handle == 10 && main.changed);
    nr_surface_mark_presented(&records[1], 2); /* This present contains only the service. */
    struct nr_surface_choice next = nr_surface_choose(records, 2, 2, 120, main.selection, 0);
    assert(next.selection.handle == 10 && !next.changed && next.age == 1);
    assert(next.reason == NR_SURFACE_CURRENT_RETAINED);
    for (uint64_t tick = 3; tick < 50; tick++) {
        nr_surface_mark_presented(&records[tick % 2], tick);
        next = nr_surface_choose(records, 2, tick, 120, next.selection, 0);
        assert(next.selection.handle == 10 && !next.changed);
    }
}

static void test_tiny_legitimate_and_unseen(void)
{
    struct nr_surface_record records[] = {
        { .handle = 11, .width = 64, .height = 32 },
        { .handle = 12, .width = 3840, .height = 2160 }
    };
    struct nr_surface_choice choice = nr_surface_choose(records, 2, 0, 120, none, 0);
    assert(choice.index == SIZE_MAX && !choice.changed);
    nr_surface_mark_presented(&records[0], 0); /* Tick zero is valid, even after wrap. */
    choice = nr_surface_choose(records, 2, 0, 120, none, 0);
    assert(choice.index == 0 && choice.selection.handle == 11);
    assert(choice.selection.width == 64 && choice.selection.height == 32 && choice.area == 2048);
    assert(choice.recent && choice.age == 0);
}

static void test_resize_destroy_and_expiration(void)
{
    struct nr_surface_record records[] = {
        { .handle = 30, .width = 1280, .height = 720, .seen = 1, .last_seen = 1 },
        { .handle = 40, .width = 160, .height = 28, .seen = 1, .last_seen = 1 }
    };
    struct nr_surface_choice choice = nr_surface_choose(records, 2, 1, 120, none, 0);
    /* The caller replaced the old live record with the new resized swapchain. */
    records[0].handle = 31;
    records[0].width = 1920; records[0].height = 1080;
    nr_surface_mark_presented(&records[0], 2);
    choice = nr_surface_choose(records, 2, 2, 120, choice.selection, 0);
    assert(choice.changed && choice.selection.handle == 31 && choice.area == 2073600);

    /* Extent changes also invalidate a selection if the driver reuses a handle. */
    records[0].width = 2560; records[0].height = 1440;
    choice = nr_surface_choose(records, 2, 3, 120, choice.selection, 0);
    assert(choice.changed && choice.selection.handle == 31);
    nr_surface_mark_presented(&records[1], 122);
    choice = nr_surface_choose(records, 2, 122, 120, choice.selection, 0);
    assert(choice.selection.handle == 31 && !choice.changed); /* Inclusive TTL boundary. */
    choice = nr_surface_choose(records, 2, 123, 120, choice.selection, 0);
    assert(choice.selection.handle == 40 && choice.changed); /* Main is now stale. */
    records[1].handle = 0; /* Caller removed the destroyed service swapchain. */
    choice = nr_surface_choose(records, 2, 123, 120, choice.selection, 0);
    assert(choice.index == SIZE_MAX && choice.selection.handle == 0 && choice.changed);
    choice = nr_surface_choose(NULL, 0, 124, 120, choice.selection, 0);
    assert(choice.index == SIZE_MAX && !choice.changed);
}

static void test_explicit_choice(void)
{
    struct nr_surface_record records[] = {
        { .handle = 50, .width = 1920, .height = 1080, .seen = 1, .last_seen = 1000 },
        { .handle = 60, .width = 64, .height = 32, .seen = 1, .last_seen = 1 }
    };
    struct nr_surface_choice choice = nr_surface_choose(records, 2, 1000, 120, none, 60);
    assert(choice.selection.handle == 60 && choice.reason == NR_SURFACE_EXPLICIT);
    assert(choice.changed && !choice.recent && !choice.preferred_unavailable);
    records[1].handle = 0;
    choice = nr_surface_choose(records, 2, 1000, 120, choice.selection, 60);
    assert(choice.selection.handle == 50 && choice.preferred_unavailable && choice.changed);
    records[1].handle = 60; records[1].seen = 0;
    choice = nr_surface_choose(records, 2, 1000, 120, choice.selection, 60);
    assert(choice.selection.handle == 60 && choice.age == UINT64_MAX && !choice.recent);
    records[1].width = 0;
    choice = nr_surface_choose(records, 2, 1000, 120, choice.selection, 60);
    assert(choice.selection.handle == 50 && choice.preferred_unavailable);
}

static void test_ties_and_record_order(void)
{
    struct nr_surface_record records[] = {
        { .handle = 80, .width = 1280, .height = 720, .seen = 1, .last_seen = 1 },
        { .handle = 70, .width = 1280, .height = 720, .seen = 1, .last_seen = 1 }
    };
    struct nr_surface_choice choice = nr_surface_choose(records, 2, 1, 120, none, 0);
    assert(choice.selection.handle == 70); /* Deterministic lowest handle. */
    struct nr_surface_record swapped[] = { records[1], records[0] };
    choice = nr_surface_choose(swapped, 2, 1, 120, choice.selection, 0);
    assert(choice.selection.handle == 70 && !choice.changed);
    struct nr_surface_selection current = { .handle = 80, .width = 1280, .height = 720 };
    choice = nr_surface_choose(records, 2, 1, 120, current, 0);
    assert(choice.selection.handle == 80 && !choice.changed);
    choice = nr_surface_choose(swapped, 2, 1, 120, current, 0);
    assert(choice.selection.handle == 80 && !choice.changed);
    records[1].width = 1920;
    choice = nr_surface_choose(records, 2, 1, 120, current, 0);
    assert(choice.selection.handle == 70 && choice.changed); /* Larger wins over current. */
}

static void test_wrap_and_area_overflow(void)
{
    struct nr_surface_record records[] = {
        { .handle = 90, .width = UINT32_MAX, .height = UINT32_MAX,
          .seen = 1, .last_seen = UINT64_MAX - 5 },
        { .handle = 100, .width = 65536, .height = 65536, .seen = 1, .last_seen = 3 }
    };
    assert(nr_surface_area(&records[0]) == UINT64_C(18446744065119617025));
    assert(nr_surface_area(&records[1]) == UINT64_C(4294967296));
    struct nr_surface_choice choice = nr_surface_choose(records, 2, 3, 9, none, 0);
    assert(choice.selection.handle == 90 && choice.age == 9 && choice.recent);
    choice = nr_surface_choose(records, 2, 3, 8, choice.selection, 0);
    assert(choice.selection.handle == 100 && choice.age == 0 && choice.changed);
    nr_surface_mark_presented(&records[0], 0);
    choice = nr_surface_choose(records, 2, 0, 0, none, 0);
    assert(choice.selection.handle == 90 && choice.age == 0);
    records[0].height = 0;
    choice = nr_surface_choose(records, 2, 0, 0, none, 0);
    assert(choice.index == SIZE_MAX); /* Future ticks from another counter are not recent. */
}

int main(void)
{
    test_main_and_service();
    test_tiny_legitimate_and_unseen();
    test_resize_destroy_and_expiration();
    test_explicit_choice();
    test_ties_and_record_order();
    test_wrap_and_area_overflow();
    puts("primary surface selection, tiny real windows, resize, expiry, ties and tick/area bounds pass (CPU)");
    return 0;
}
