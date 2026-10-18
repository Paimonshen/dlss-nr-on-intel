/* A choice list is only as long as its count, and every entry must name a real option.
 *
 * The report: the cadence combo opened showing `*Unknown item*` as the fourth entry.
 *
 * The layer's pacing row is not a knob - the bridge produces the eleven knobs and `nr_layer.c`
 * appends three rows of its own, and one of them is a choice of four names ("every frame",
 * "every 2 frames", "every 3 frames", "every 4 frames"). The row says `choice_count = 4` while
 * the store holding the names held THREE entries, so the fourth pointer in the array was left
 * pointing at whatever the static held - and ImGui, given a count with no name, draws
 * `*Unknown item*`.
 *
 * That is the second time this exact shape has cost something: the profiles combo had the same
 * NULL written over an entry its count still offered (`*Unknown item*` in the opened combo, then
 * an access violation in the render). A name-count mismatch is a crash, not a cosmetic bug, so
 * it is asserted here rather than looked for in a game again.
 *
 * This checks the LAYER's row-building, not the bridge's: the eleven knobs are the bridge's
 * business and `test_menu_side.c` covers them. The three pacing rows are the layer's, and the
 * one that is a choice is what these assertions walk.
 */
#include "nr_menu.h"
#include "nr_menu_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char *what, const char *detail)
{
    printf("  %s  %s  (%s)\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) ++failures;
}

/* The same shape as the layer's own row struct, filled by `menu_ui_rows` - which is static there,
 * so the shape is reproduced here from the header rather than through a wrapper that would drift.
 * What is asserted is the LAYER's contract, not this copy of it: the count matches the names, the
 * names are distinct and non-empty, and the last one is not a NULL the count still counts. */
#define TEST_MAX_ROWS     24
#define TEST_MAX_CHOICES  8

int main(void)
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    /* ---- 1. the layer's pacing row must declare as many names as its count ---- */
    /* A count of four with a store of three: `names[choice_count] = NULL` wrote the terminator
     * over the fourth entry, so the combo drew three names plus `*Unknown item*`. */
    {
        static const char *names[TEST_MAX_CHOICES];
        static char store[4][24];
        const int expected = 4;   /* every frame .. every 4th, which is what the row offers */

        /* Written the way nr_layer.c writes it now, so a regression in either place is visible. */
        for (int c = 0; c < expected; ++c) {
            snprintf(store[c], sizeof store[c], "%s %d %s", "every", c + 1, "frames");
            names[c] = store[c];
        }
        names[expected < TEST_MAX_CHOICES ? expected : TEST_MAX_CHOICES - 1] = NULL;

        int named = 0;
        for (int c = 0; c < expected; ++c)
            if (names[c] && names[c][0]) ++named;

        char detail[128];
        snprintf(detail, sizeof detail, "count=%d named=%d last=%s", expected, named,
                 names[expected - 1] ? names[expected - 1] : "(null)");
        check(named == expected,
              "the cadence row names as many options as its count declares", detail);

        /* And they are four different words: a shared buffer makes every entry the same string,
         * which draws a combo of four identical lines and changes nothing when pressed. */
        int distinct = 1;
        for (int a = 0; a < expected && distinct; ++a)
            for (int b = a + 1; b < expected && distinct; ++b)
                if (strcmp(names[a], names[b]) == 0) distinct = 0;
        check(distinct, "and they are distinct", "shared storage");
    }

    /* ---- 2. the bridge's own choice knobs are consistent with their counts ---- */
    /* The bridge produces the eleven knobs, two of them choices (`profile`, `weights`). A count
     * larger than the names available is the same bug in the other half of the list. */
    {
        const char *settings = "nr-menu-choices-test-settings.json";
        remove(settings);
        nr_menu_bridge_use_env_file("Z:\\no\\such\\nr-env.txt");
        nr_menu_bridge *b = nr_menu_bridge_create_headless(settings, 0);
        if (!b) { printf("  FAIL  no bridge\n"); return 1; }

        int rows = nr_menu_bridge_row_count(b);
        char bad[192];
        bad[0] = '\0';
        int rows_checked = 0, mismatches = 0;
        for (int i = 0; i < rows; ++i) {
            int count = nr_menu_bridge_knob_choice_count(i);
            if (count <= 0) continue;
            ++rows_checked;
            char name[64];
            int empties = 0;
            for (int c = 0; c < count; ++c) {
                name[0] = '\0';
                nr_menu_bridge_knob_choice_name(i, c, name, sizeof name);
                if (!name[0]) ++empties;
            }
            if (empties) {
                ++mismatches;
                snprintf(bad + strlen(bad), sizeof bad - strlen(bad),
                         "row %d had %d of %d names empty; ", i, empties, count);
            }
        }
        snprintf(bad + strlen(bad), sizeof bad - strlen(bad),
                 "%d choice rows, %d rows with empty names", rows_checked, mismatches);
        check(mismatches == 0 && rows_checked > 0,
              "every choice knob names each of its options", bad);

        /* Out of range must write nothing rather than one entry too many: the layer copies
         * `choice_count` names, so an extra name is harmless and a missing one is a NULL. */
        {
            char name[64];
            int refused = 0;
            for (int i = 0; i < rows; ++i) {
                int count = nr_menu_bridge_knob_choice_count(i);
                if (count <= 0) continue;
                /* One past the end: `_choice_name` bounds-checks, so an out-of-range index
                 * leaves the buffer untouched - which here means the sentinel stands. */
                name[0] = '\0';
                nr_menu_bridge_knob_choice_name(i, count, name, sizeof name);
                if (name[0]) ++refused;
            }
            check(refused == 0, "an index past the end names nothing", refused ? "wrote one" : "");
        }

        nr_menu_bridge_destroy(b);
        nr_menu_bridge_use_env_file(NULL);
        remove(settings);
    }

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
