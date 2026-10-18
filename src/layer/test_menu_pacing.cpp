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

    nr_menu_bridge_destroy(b);
    nr_menu_bridge_use_env_file(NULL);
    system("rmdir /s /q nr-pacing-test 2>nul");
    system("rmdir /s /q nr-pacing-test-odd 2>nul");
    remove("nr-pacing-test\\work\\nr_settings.json");
    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
