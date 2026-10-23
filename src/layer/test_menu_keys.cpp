/* The panel's hotkeys, checked where they cannot be faked.
 *
 * The report was "S does not save, R does not reset, END does nothing" while the panel drew and
 * its buttons worked. The cause was structural: all three lived inside `nr_menu_update`'s
 * `if (m->open && m->read_keys)` block, and the ImGui build sets `read_keys` to 0 - so the block
 * never ran. Two earlier attempts at a test missed it, and both are recorded here because each
 * looked like it worked:
 *
 *   1. Driving `nr_menu_bridge_save` / `_reset` passes whatever the key code does: those entry
 *      points do not go through the key reading at all.
 *   2. Driving a real `nr_menu` needs a Vulkan device - `nr_menu_create` asks the driver for
 *      memory properties - so a headless test cannot reach `nr_menu_update` with `read_keys` off.
 *
 * What CAN be checked without a device is the structure that was wrong: that each hotkey's read
 * is outside the `read_keys` guard, and that the effect toggle is Home rather than END. That is a
 * source check, so it fails the moment somebody moves one back inside the block - which is the
 * regression this file exists for.
 */
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char *what, const char *detail)
{
    printf("  %s  %s  (%s)\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) ++failures;
}

/* The text between the start of `nr_menu_update` and the end of its `read_keys` block, plus the
 * text before it. Split there so a hotkey can be asserted to be in the right half. */
static int find(const char *haystack, const char *needle)
{
    return strstr(haystack, needle) != NULL;
}

int main(void)
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    FILE *f = fopen("src/layer/nr_menu.c", "rb");
    if (!f) {
        printf("  FAIL  nr_menu.c was not readable  (run from the repository root)\n");
        return 1;
    }
    static char source[200000];
    size_t n = fread(source, 1, sizeof source - 1, f);
    source[n] = '\0';
    fclose(f);

    printf("the panel's hotkeys, by structure\n");

    /* The block that was dead. Everything between this marker and its closing brace is skipped
     * when `read_keys` is 0, so nothing a player must be able to press may live there. */
    const char *guard = strstr(source, "if (m->open && m->read_keys) {");
    if (!guard) {
        printf("  FAIL  the read_keys guard was not found  (has nr_menu_update been restructured?)\n");
        return 1;
    }

    /* The S and R reads, which the fix moved above the guard. */
    const char *save = strstr(source, "if (key_pressed(m, NR_KEY_S)) m->save_pressed = 1;");
    const char *reset = strstr(source, "if (key_pressed(m, NR_KEY_R)) m->reset_pressed = 1;");

    check(save != NULL, "S is still read somewhere", save ? "found" : "NOT FOUND");
    check(save != NULL && save < guard,
          "S is read OUTSIDE the read_keys block",
          save && save < guard ? "above the guard" : "inside the guard - S is dead in ImGui builds");

    check(reset != NULL, "R is still read somewhere", reset ? "found" : "NOT FOUND");
    check(reset != NULL && reset < guard,
          "R is read OUTSIDE the read_keys block",
          reset && reset < guard ? "above the guard" : "inside the guard - R is dead in ImGui builds");

    /* The effect toggle is Home, 0x24, and no longer END, 0x23. END is bound by enough games
     * that the key never arrived; the report was "END does nothing". */
    check(find(source, "#define NR_KEY_TOGGLE  0x24"),
          "the effect toggle is Home (0x24)",
          find(source, "#define NR_KEY_TOGGLE  0x24") ? "0x24" : "not 0x24 - see the define");
    check(!find(source, "#define NR_KEY_COMPARE"),
          "the comparison key is gone",
          find(source, "#define NR_KEY_COMPARE") ? "NR_KEY_COMPARE still defined" : "removed");

    /* The comparison feature itself: no compare-mode transitions left in the key handling. This
     * is what "delete the Home function" means - the key moved to the toggle, so any remaining
     * compare_mode bump on a key would be a second meaning on one key. */
    check(!find(source, "if (key_pressed(m, NR_KEY_C))"),
          "no key still cycles the comparison",
          find(source, "if (key_pressed(m, NR_KEY_C))") ? "NR_KEY_C still handled" : "removed");

    /* The layer reads Home, S and R itself for the ImGui build, next to INS and L. */
    FILE *layer = fopen("src/layer/nr_layer.c", "rb");
    if (!layer) {
        printf("  FAIL  nr_layer.c was not readable\n");
        return 1;
    }
    static char lsource[400000];
    size_t ln = fread(lsource, 1, sizeof lsource - 1, layer);
    lsource[ln] = '\0';
    fclose(layer);

    check(find(lsource, "nr_menu_inject_toggle(data->menu)"),
          "the layer raises the toggle for the ImGui build",
          find(lsource, "nr_menu_inject_toggle(data->menu)") ? "present" : "MISSING");
    check(find(lsource, "nr_menu_inject_save(data->menu)"),
          "the layer raises Save for the ImGui build",
          find(lsource, "nr_menu_inject_save(data->menu)") ? "present" : "MISSING");
    check(find(lsource, "nr_menu_inject_reset(data->menu)"),
          "the layer raises Reset for the ImGui build",
          find(lsource, "nr_menu_inject_reset(data->menu)") ? "present" : "MISSING");
    check(find(lsource, "GetAsyncKeyState(0x24)"),
          "and reads Home itself (0x24)",
          find(lsource, "GetAsyncKeyState(0x24)") ? "0x24" : "MISSING");

    /* ---- the bridge must not consume the toggle ----
     *
     * `nr_menu_bridge_update` runs BEFORE the layer asks the menu for the toggle, and
     * `nr_menu_take_toggle_request` clears the flag. So a bridge that takes it leaves the layer
     * with nothing: END did nothing, and then Home did nothing, for exactly this reason while
     * every panel button worked. The bridge has no way to act on the request - the trigger file
     * is the layer's - so it must leave it alone. */
    FILE *bridge_src = fopen("src/layer/nr_menu_bridge.c", "rb");
    if (!bridge_src) {
        printf("  FAIL  nr_menu_bridge.c was not readable\n");
        return 1;
    }
    static char bsource[400000];
    size_t bn = fread(bsource, 1, sizeof bsource - 1, bridge_src);
    bsource[bn] = '\0';
    fclose(bridge_src);

    check(!find(bsource, "if (nr_menu_take_toggle_request(b->menu))"),
          "the bridge does NOT consume the toggle request",
          find(bsource, "if (nr_menu_take_toggle_request(b->menu))")
              ? "still consumed - the layer then sees nothing and the key does nothing"
              : "left for the layer, which owns the trigger");

    /* ---- and the panel must not let nav eat Home ----
     *
     * With the panel open, ImGui treats Home as "focus the first item", so the key moved the
     * selection instead of reaching the toggle. Claiming it for the window is what stops nav. */
    FILE *ui_src = fopen("src/layer/nr_menu_ui.cpp", "rb");
    if (!ui_src) {
        printf("  FAIL  nr_menu_ui.cpp was not readable\n");
        return 1;
    }
    static char usource[400000];
    size_t un = fread(usource, 1, sizeof usource - 1, ui_src);
    usource[un] = '\0';
    fclose(ui_src);

    check(find(usource, "ImGui::SetKeyOwner(ImGuiKey_Home"),
          "the panel claims Home so nav cannot act on it",
          find(usource, "ImGui::SetKeyOwner(ImGuiKey_Home)")
              ? "claimed"
              : "not claimed - Home will move the selection instead of toggling the effect");

    /* ---- the C/C++ boundary is 32-bit safe ----
     *
     * Dead or Alive 5 is a 32-bit DirectX 9 game, and it crashed with the panel open about four
     * seconds in, right after DXVK creates the swapchain, at an address in no module. Two defects,
     * both invisible on x64 where every calling convention is the same and every handle is a
     * pointer:
     *
     *   1. The GPA typedef had no `VKAPI_PTR`. On x86 Vulkan's entry points are `__stdcall`, so
     *      every resolve unbalanced the stack by the two arguments it never cleaned up - a
     *      corrupted return address, which is what a wild `ret` address looks like.
     *   2. `nrex_image` / `nrex_image_view` returned `void *`. Vulkan's non-dispatchable handles are
     *      `uint64_t` on 32-bit builds (the macro keys on `_WIN64`/`__LP64__`), so a pointer return
     *      dropped the top half and the blend pass was handed a truncated handle.
     *
     * A third one the second exposed: the loader context was `void *ctx[]`, and two of its four
     * slots are `VkDevice`/`VkInstance`, which are 64 bits there too.
     *
     * These are read from the source because the defect is a TYPE, and a type is not something a
     * running test can observe on the platform where it is harmless. The assertions are negative
     * on purpose: they fail if someone reintroduces the shape. */
    check(find(usource, "VKAPI_PTR *nrex_gpa_device_fn"),
          "the device resolver carries VKAPI_PTR",
          find(usource, "VKAPI_PTR *nrex_gpa_device_fn")
              ? "__stdcall on x86" : "MISSING - the stack unbalances by 8 bytes per call on 32-bit");
    check(find(usource, "VKAPI_PTR *nrex_gpa_instance_fn"),
          "the instance resolver carries VKAPI_PTR",
          find(usource, "VKAPI_PTR *nrex_gpa_instance_fn")
              ? "__stdcall on x86" : "MISSING");
    check(find(usource, "uint64_t nrex_image(nrex_ui *ui)"),
          "nrex_image returns a 64-bit handle",
          find(usource, "uint64_t nrex_image(nrex_ui *ui)")
              ? "uint64_t" : "MISSING - a void* return truncates VkImage on x86");
    check(find(usource, "uint64_t nrex_image_view(nrex_ui *ui)"),
          "nrex_image_view returns a 64-bit handle",
          find(usource, "uint64_t nrex_image_view(nrex_ui *ui)")
              ? "uint64_t" : "MISSING - a void* return truncates VkImageView on x86");
    check(!find(usource, "void **ctx = (void **)user"),
          "the loader context is not void* slots",
          find(usource, "void **ctx = (void **)user")
              ? "back to void* - VkDevice and VkInstance are truncated on x86"
              : "uint64_t slots");
    check(find(usource, "uint64_t *ctx = (uint64_t *)user"),
          "the loader context uses 64-bit slots",
          find(usource, "uint64_t *ctx = (uint64_t *)user")
              ? "uint64_t slots" : "MISSING");

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
