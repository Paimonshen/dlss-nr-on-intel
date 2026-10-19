/* The layer's own pacing: three settings that are not knobs, and the conflicts between them.
 *
 * Asynchronous mode, the GPU image path, and how many presents each result covers live in the
 * proxy's `nr-env.txt` rather than the settings file, because the LAYER reads them once at
 * start-up while the daemon reads the settings file every frame. So the panel writes that file,
 * and the setup window writes the same one - which is why the conflicts have to be applied in one
 * place and not duplicated per caller.
 *
 * The rule the setup window states, and the one asserted here: the GPU image path requires
 * asynchronous mode OFF and every frame processed. Turning the path on must therefore clear both,
 * and turning either of those on must clear the path. A panel that let all three be true at once
 * would silently skip the effect, because the layer refuses that combination.
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

/* A whole installation, as far as this is concerned: `work/nr_settings.json` beside
 * `dlss-nr/nr-env.txt`, which is the layout `env_file_path` derives. */
static int make_installation(const char *root, const char *env_body)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "mkdir \"%s\\work\" 2>nul & mkdir \"%s\\dlss-nr\" 2>nul", root, root);
    if (system(cmd) != 0) { /* the folders may already exist; the files below decide */ }

    char path[1024];
    snprintf(path, sizeof path, "%s\\work\\nr_settings.json", root);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fputs("{\n  \"render_scale\": 0.8\n}\n", f);
    fclose(f);

    snprintf(path, sizeof path, "%s\\dlss-nr\\nr-env.txt", root);
    f = fopen(path, "wb");
    if (!f) return 0;
    fputs(env_body, f);
    fclose(f);
    return 1;
}

static int env_line(const char *root, const char *key, char *out, size_t size)
{
    char path[1024];
    snprintf(path, sizeof path, "%s\\dlss-nr\\nr-env.txt", root);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    static char text[8192];
    size_t n = fread(text, 1, sizeof text - 1, f);
    text[n] = '\0';
    fclose(f);
    char needle[96];
    snprintf(needle, sizeof needle, "%s=", key);
    const char *at = strstr(text, needle);
    if (!at) return 0;
    const char *p = at + strlen(needle);
    size_t i = 0;
    while (*p && *p != '\n' && *p != '\r' && i + 1 < size) out[i++] = *p++;
    out[i] = '\0';
    return 1;
}

static int env_mentions_count(const char *root, const char *key)
{
    char path[1024];
    snprintf(path, sizeof path, "%s\\dlss-nr\\nr-env.txt", root);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    static char text[8192];
    size_t n = fread(text, 1, sizeof text - 1, f);
    text[n] = '\0';
    fclose(f);
    char needle[96];
    snprintf(needle, sizeof needle, "%s=", key);
    int count = 0;
    for (const char *p = text; (p = strstr(p, needle)) != NULL; ++p) ++count;
    return count;
}

int main(void)
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    printf("the layer's pacing, and the conflicts between the three\n");

    const char *root = "nr-pacing-test";
    /* An untouched install: the setup window's defaults, plus a line the panel must not lose. */
    const char *body =
        "# Set by the setup window.\n"
        "NR_ROOT=C:\\somewhere\\release\n"
        "NR_LAYER_ASYNC=0\n"
        "NR_GPU_IMAGE=0\n"
        "NR_GPU_TRANSPORT=0\n"
        "NR_GPU_ONLY=0\n"
        "NR_LAYER_LIVE=1\n"
        "NR_PYTHON=C:\\python.exe\n";
    if (!make_installation(root, body)) { printf("  FAIL  could not build the fixture\n"); return 1; }

    /* The production path is the `nr-env.txt` beside the layer DLL. A fixture has no installation
     * around it, so it names the file it just built. */
    char env_path[1024];
    snprintf(env_path, sizeof env_path, "%s\\dlss-nr\\nr-env.txt", root);
    nr_menu_bridge_use_env_file(env_path);

    char settings[1024];
    snprintf(settings, sizeof settings, "%s\\work\\nr_settings.json", root);
    nr_menu_bridge *b = nr_menu_bridge_create_headless(settings, 0);
    if (!b) { printf("  FAIL  no bridge\n"); return 1; }

    char detail[192];
    char value[32];

    /* ---- the file is found and read ---- */
    check(nr_menu_bridge_pacing_available(b), "the environment file is located",
          nr_menu_bridge_pacing_available(b) ? "available" : "NOT FOUND - the panel would offer nothing");

    snprintf(detail, sizeof detail, "async=%d gpu=%d every=%d",
             nr_menu_bridge_pacing_async(b), nr_menu_bridge_pacing_gpu_image(b),
             nr_menu_bridge_pacing_every(b));
    check(!nr_menu_bridge_pacing_async(b) && !nr_menu_bridge_pacing_gpu_image(b) &&
          nr_menu_bridge_pacing_every(b) == 1,
          "and its values are what the panel starts from", detail);

    /* ---- turning the GPU path on clears the other two ---- */
    {
        nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 1);
        nr_menu_bridge_set_pacing(b, NR_PACING_EVERY, 3);
        snprintf(detail, sizeof detail, "before: async=%d every=%d",
                 nr_menu_bridge_pacing_async(b), nr_menu_bridge_pacing_every(b));
        check(nr_menu_bridge_pacing_async(b) && nr_menu_bridge_pacing_every(b) == 3,
              "asynchronous and a slower cadence can be set", detail);

        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 1);
        snprintf(detail, sizeof detail, "after: gpu=%d async=%d every=%d",
                 nr_menu_bridge_pacing_gpu_image(b), nr_menu_bridge_pacing_async(b),
                 nr_menu_bridge_pacing_every(b));
        /* The path requires both, so both are cleared - and this is the conflict the setup
         * window's note describes rather than a preference. */
        check(nr_menu_bridge_pacing_gpu_image(b) == 1 &&
              nr_menu_bridge_pacing_async(b) == 0 &&
              nr_menu_bridge_pacing_every(b) == 1,
              "turning the GPU path on clears asynchronous mode and the cadence", detail);
    }

    /* ---- and each of the other two clears the path ---- */
    {
        nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 1);
        snprintf(detail, sizeof detail, "gpu=%d after turning asynchronous on",
                 nr_menu_bridge_pacing_gpu_image(b));
        check(nr_menu_bridge_pacing_gpu_image(b) == 0,
              "turning asynchronous mode on clears the GPU path", detail);

        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 1);
        nr_menu_bridge_set_pacing(b, NR_PACING_EVERY, 2);
        snprintf(detail, sizeof detail, "gpu=%d after a slower cadence",
                 nr_menu_bridge_pacing_gpu_image(b));
        check(nr_menu_bridge_pacing_gpu_image(b) == 0,
              "choosing fewer than every frame clears the GPU path", detail);
    }

    /* ---- Save and Reset do not govern the pacing rows, and the rows do not wait for Save ---- */
    {
        /* The three rows are the LAYER's start-up settings and live in `nr-env.txt`; Save and
         * Reset are the settings file's, which is the daemon's. These checks pin the separation
         * down in both directions, because it is the thing a reader is most likely to assume
         * otherwise: "a panel with a Save button must save everything" is a reasonable guess and
         * it is wrong here.
         *
         * Direction 1: moving a row writes the file on its own. Nothing is saved afterwards.
         * Direction 2: Save and Reset leave that file byte for byte alone. */
        nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 0);
        nr_menu_bridge_set_pacing(b, NR_PACING_EVERY, 4);

        char path[1024];
        snprintf(path, sizeof path, "%s\\dlss-nr\\nr-env.txt", root);
        char before[8192] = "";
        FILE *f = fopen(path, "rb");
        if (!f) { printf("  FAIL  the environment file vanished\n"); return 1; }
        size_t n = fread(before, 1, sizeof before - 1, f);
        before[n] = 0;
        fclose(f);

        snprintf(detail, sizeof detail, "NR_LAYER_ASYNC=%s, NR_LAYER_LIVE=%s",
                 env_line(root, "NR_LAYER_ASYNC", value, sizeof value) ? value : "?",
                 env_line(root, "NR_LAYER_LIVE", value, sizeof value) ? value : "?");
        check(strstr(before, "NR_LAYER_ASYNC=0") && strstr(before, "NR_LAYER_LIVE=4"),
              "a row moved writes the file with no Save pressed", detail);

        /* Reset has two paths and both must leave the pacing alone. This is the headless one
         * (`nr_menu_bridge_reset`, taken when there is no menu); the menu's own is the
         * `nr_menu_take_reset` branch in `nr_menu_bridge_update`, which runs the same body -
         * `load_defaults`, the moved flags, the preview - and likewise writes no pacing.
         *
         * The preview is what proves Reset ran at all. Without that, "the pacing file did not
         * change" would also pass for a Reset that never happened, which is the shape of check
         * that passes for the wrong reason - mutation testing the reset branch caught exactly
         * that, the file check going green because nothing had run. */
        char preview[1024];
        snprintf(preview, sizeof preview, "%s.live", settings);
        remove(preview);
        nr_menu_bridge_reset(b);
        f = fopen(preview, "rb");
        check(f != NULL, "Reset wrote the preview, so it ran", f ? "preview present" : "NO PREVIEW");
        if (f) fclose(f);
        char after_reset[8192] = "";
        f = fopen(path, "rb");
        n = f ? fread(after_reset, 1, sizeof after_reset - 1, f) : 0;
        after_reset[n] = 0;
        if (f) fclose(f);
        check(strcmp(before, after_reset) == 0, "Reset leaves the pacing file byte for byte",
              strcmp(before, after_reset) == 0 ? "unchanged" : "REWRITTEN");

        /* Save: it writes the settings file and the preview, nothing of the pacing. */
        nr_menu_bridge_save(b);
        char after_save[8192] = "";
        f = fopen(path, "rb");
        n = f ? fread(after_save, 1, sizeof after_save - 1, f) : 0;
        after_save[n] = 0;
        if (f) fclose(f);
        check(strcmp(before, after_save) == 0, "Save leaves the pacing file byte for byte",
              strcmp(before, after_save) == 0 ? "unchanged" : "REWRITTEN");

        char saved[1024];
        snprintf(saved, sizeof saved, "%s\\work\\nr_settings.json", root);
        f = fopen(saved, "rb");
        check(f != NULL, "and Save is what leaves the settings file", f ? "written" : "MISSING");
        if (f) fclose(f);

        /* Reset must not have reached the pacing values either, on screen or in the bridge. */
        snprintf(detail, sizeof detail, "async=%d every=%d after Reset",
                 nr_menu_bridge_pacing_async(b), nr_menu_bridge_pacing_every(b));
        check(nr_menu_bridge_pacing_async(b) == 0 && nr_menu_bridge_pacing_every(b) == 4,
              "and Reset does not reset the rows it does not own", detail);
    }

    /* ---- what the four variables mean, and one of them is NOT the panel's ---- */
    {
        /* `NR_GPU_ONLY` is the STRICT half of the GPU path: with it on, the layer refuses the CPU
         * pixel fallback and keeps the game's own image. The setup window writes it always as
         * "0" and says why - "Never strict: where a game cannot keep its frames on the GPU, NR
         * takes its usual path and the layer's line in the log says why, where the strict mode
         * left the game's frame alone and said nothing a user could see" (issue #17).
         *
         * The panel used to write it as a copy of the GPU image path, so turning that row on also
         * turned strict mode on: in a game whose shared path cannot run, the user got the game's
         * own frame and no effect at all - the exact failure issue #17 removed. This is the
         * regression check for that, and it fails if the two are ever tied together again. */
        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 1);
        char strict[32] = "";
        int have = env_line(root, "NR_GPU_ONLY", strict, sizeof strict);
        snprintf(detail, sizeof detail, "NR_GPU_ONLY=%s beside NR_GPU_IMAGE=1",
                 have ? strict : "absent");
        check(have && strcmp(strict, "0") == 0,
              "the GPU path does not turn strict mode on with it", detail);
        /* Still written, not omitted: an install that had strict mode on is cleared by this. */
        check(have, "and strict mode is written as off rather than left alone", detail);
    }

    /* ---- the file is rewritten, and only those lines ---- */
    {
        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 1);
        check(env_line(root, "NR_GPU_IMAGE", value, sizeof value) && strcmp(value, "1") == 0,
              "the written file says the path is on", value);
        check(env_line(root, "NR_LAYER_ASYNC", value, sizeof value) && strcmp(value, "0") == 0,
              "and that asynchronous mode is off", value);
        check(env_line(root, "NR_LAYER_LIVE", value, sizeof value) && strcmp(value, "1") == 0,
              "and that the cadence is every frame", value);

        /* One line each: the substitute-in-place and the append must not both run. */
        char counts[96];
        snprintf(counts, sizeof counts, "GPU_IMAGE x%d, ASYNC x%d, LIVE x%d",
                 env_mentions_count(root, "NR_GPU_IMAGE"),
                 env_mentions_count(root, "NR_LAYER_ASYNC"),
                 env_mentions_count(root, "NR_LAYER_LIVE"));
        check(env_mentions_count(root, "NR_GPU_IMAGE") == 1 &&
              env_mentions_count(root, "NR_LAYER_ASYNC") == 1 &&
              env_mentions_count(root, "NR_LAYER_LIVE") == 1,
              "and each appears exactly once", counts);

        /* Every line the panel does not own is still there, byte for byte. */
        check(env_line(root, "NR_PYTHON", value, sizeof value) &&
              strcmp(value, "C:\\python.exe") == 0,
              "the lines the panel does not own survive", value);
        check(env_line(root, "NR_ROOT", value, sizeof value) &&
              strcmp(value, "C:\\somewhere\\release") == 0,
              "including the release root", value);

        FILE *f = fopen("nr-pacing-test\\dlss-nr\\nr-env.txt", "rb");
        int kept_comment = 0;
        if (f) {
            static char text[8192];
            size_t n = fread(text, 1, sizeof text - 1, f);
            text[n] = '\0';
            fclose(f);
            kept_comment = strstr(text, "# Set by the setup window.") != NULL;
        }
        check(kept_comment, "and the header comment", kept_comment ? "kept" : "LOST");
    }

    /* ---- round trip: what the panel wrote is what the next start reads ---- */
    {
        /* This is the start-up half of the same question, and it is the one that cannot be
         * answered by looking at the writer: a value written in a shape the reader does not
         * recognise would be a setting that silently reverts at the next launch. So write the
         * rows, throw the bridge away, and build a new one over the same files - which is what a
         * game restart does.
         *
         * The cadence is the interesting one (1-4 becomes the text the row shows and back), and
         * the conflicts are the pair that could be written inconsistently by a half-applied
         * change. */
        nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 0);
        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 0);
        nr_menu_bridge_set_pacing(b, NR_PACING_EVERY, 3);

        nr_menu_bridge_destroy(b);
        b = nr_menu_bridge_create_headless(settings, 0);
        if (!b) { printf("  FAIL  no bridge on the restart\n"); return 1; }
        snprintf(detail, sizeof detail, "after a restart: async=%d gpu=%d every=%d",
                 nr_menu_bridge_pacing_async(b), nr_menu_bridge_pacing_gpu_image(b),
                 nr_menu_bridge_pacing_every(b));
        check(!nr_menu_bridge_pacing_async(b) && !nr_menu_bridge_pacing_gpu_image(b) &&
              nr_menu_bridge_pacing_every(b) == 3,
              "a restart reads back the cadence that was written", detail);

        nr_menu_bridge_set_pacing(b, NR_PACING_GPU_IMAGE, 1);
        nr_menu_bridge_destroy(b);
        b = nr_menu_bridge_create_headless(settings, 0);
        if (!b) { printf("  FAIL  no bridge after the GPU path restart\n"); return 1; }
        snprintf(detail, sizeof detail, "after a restart: async=%d gpu=%d every=%d",
                 nr_menu_bridge_pacing_async(b), nr_menu_bridge_pacing_gpu_image(b),
                 nr_menu_bridge_pacing_every(b));
        check(nr_menu_bridge_pacing_gpu_image(b) == 1 && !nr_menu_bridge_pacing_async(b) &&
              nr_menu_bridge_pacing_every(b) == 1,
              "and a restart reads back the GPU path with the two it cleared", detail);
    }

    /* ---- an install that is not shaped as expected offers nothing ---- */
    {
        nr_menu_bridge_destroy(b);
        const char *odd = "nr-pacing-test-odd";
        char path[1024];
        snprintf(path, sizeof path, "%s", odd);
        (void)path;
        /* A settings file with no `work/` parent: `env_file_path` refuses, and the panel must
         * then not offer the three rows at all. */
        nr_menu_bridge_use_env_file("Z:\\no\\such\\nr-env.txt");
        b = nr_menu_bridge_create_headless("nr-pacing-test-odd/settings.json", 0);
        if (!b) { printf("  FAIL  no bridge on the odd shape\n"); return 1; }
        snprintf(detail, sizeof detail, "available=%d", nr_menu_bridge_pacing_available(b));
        check(!nr_menu_bridge_pacing_available(b),
              "an install of an unexpected shape offers no pacing rows", detail);

        int wrote = nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 1);
        snprintf(detail, sizeof detail, "set_pacing returned %d", wrote);
        check(wrote == 0, "and refuses to write", detail);
    }

    /* ---- the pacing rows do not depend on the settings file ---- */
    {
        /* They live in `nr-env.txt`; the settings file is the daemon's. Reading one inside the
         * other's parse branch made them die together, and the consequence was visible: a
         * settings file that was absent or empty left the panel offering no pacing rows at all,
         * with `set_pacing` refusing to write - so the three settings that decide how the
         * network is fed became unreachable because an unrelated JSON file was missing. */
        nr_menu_bridge_destroy(b);
        const char *bare = "nr-pacing-test-bare";
        char cmd[1024];
        snprintf(cmd, sizeof cmd, "mkdir \"%s\\dlss-nr\" 2>nul & mkdir \"%s\\work\" 2>nul", bare, bare);
        if (system(cmd) != 0) { /* the folders may already exist */ }
        char path[1024];
        snprintf(path, sizeof path, "%s\\dlss-nr\\nr-env.txt", bare);
        FILE *f = fopen(path, "wb");
        fputs("NR_ROOT=C:\\somewhere\\release\n"
              "NR_LAYER_ASYNC=1\n"
              "NR_GPU_IMAGE=0\n"
              "NR_GPU_TRANSPORT=0\n"
              "NR_GPU_ONLY=0\n"
              "NR_LAYER_LIVE=3\n", f);
        fclose(f);

        nr_menu_bridge_use_env_file(path);
        char bare_settings[1024];
        snprintf(bare_settings, sizeof bare_settings, "%s\\work\\nr_settings.json", bare);

        /* No settings file at all. */
        b = nr_menu_bridge_create_headless(bare_settings, 0);
        if (!b) { printf("  FAIL  no bridge on the bare shape\n"); return 1; }
        snprintf(detail, sizeof detail, "available=%d async=%d every=%d",
                 nr_menu_bridge_pacing_available(b), nr_menu_bridge_pacing_async(b),
                 nr_menu_bridge_pacing_every(b));
        check(nr_menu_bridge_pacing_available(b) && nr_menu_bridge_pacing_async(b) &&
              nr_menu_bridge_pacing_every(b) == 3,
              "a missing settings file leaves the pacing rows readable", detail);
        int wrote = nr_menu_bridge_set_pacing(b, NR_PACING_ASYNC, 0);
        snprintf(detail, sizeof detail, "set_pacing returned %d", wrote);
        check(wrote == 1, "and writable", detail);

        /* And an empty one: present, no JSON in it, still nothing to do with pacing. */
        f = fopen(bare_settings, "wb");
        fclose(f);
        nr_menu_bridge_destroy(b);
        b = nr_menu_bridge_create_headless(bare_settings, 0);
        if (!b) { printf("  FAIL  no bridge on the empty settings shape\n"); return 1; }
        snprintf(detail, sizeof detail, "available=%d", nr_menu_bridge_pacing_available(b));
        check(nr_menu_bridge_pacing_available(b),
              "an empty settings file leaves them readable too", detail);
    }

    nr_menu_bridge_destroy(b);
    nr_menu_bridge_use_env_file(NULL);
    system("rmdir /s /q nr-pacing-test 2>nul");
    system("rmdir /s /q nr-pacing-test-odd 2>nul");
    system("rmdir /s /q nr-pacing-test-bare 2>nul");
    remove("nr-pacing-test\\work\\nr_settings.json");
    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
