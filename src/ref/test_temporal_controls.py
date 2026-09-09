#!/usr/bin/env python3
"""The temporal controls the vendor's panel exposes, and the two HANDOFF called missing.

Scene-cut detection and per-pixel history confidence were both listed as still to be
built. They were already in the recovered pipeline; this exercises them, and the motion
scale that the panel calls Motion Scale X/Y Multiplier.
"""
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ref"))
import image_io, nr_frame, nr_model, nr_temporal  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def main():
    source = image_io.load(str(ROOT / "pngs" / "Cyberpunk-2077_02.jpg"))
    frames, motions = nr_temporal.pan_sequence(source, shift=(6, 0), size=(192, 192), frames=4)
    model = nr_frame.ResidentBackend()
    nr_temporal.install_gpu_history(model.runtime)

    _, plain = nr_temporal.run_sequence(model, frames, motions, verbose=False)
    settled = plain[-1]["alpha_mean"]
    check("history raises the learned blend", plain[0]["alpha_mean"] < 0.1 < settled,
          f"{plain[0]['alpha_mean']:.4f} with none, {settled:.4f} settled")

    # One transition, not two: the first half pans across one scene and the second
    # half across another, so there is exactly one luma step to find.
    other = image_io.load(str(ROOT / "pngs" / "Cyberpunk-2077_01.jpg"))
    second, _ = nr_temporal.pan_sequence(other, shift=(6, 0), size=(192, 192), frames=2)
    cut_frames = frames[:2] + second
    luma_step = float(np.abs(cut_frames[2].mean(axis=2) - cut_frames[1].mean(axis=2)).mean())

    check("the cut frame is a real luma step", luma_step > 0.05, f"mean |dY| {luma_step:.4f}")

    def sequence_with(threshold):
        """Run the cut sequence and return (the session, its per-frame alphas)."""
        live = nr_temporal.session(model, motion="zero", scene_cut_threshold=threshold)
        geometry = nr_frame.NetworkGeometry.vendor_aligned(192, 192)
        alphas = []
        for index, frame in enumerate(cut_frames):
            live.process(frame, motion=None if index == 0 else motions[index])
            alphas.append(float(geometry.crop(
                nr_temporal.blend_alpha(live.pipeline.head)).mean()))
        return live, alphas

    detected, detected_alpha = sequence_with(0.05)
    ignored, ignored_alpha = sequence_with(0.0)
    check("the detector counts exactly the one cut", detected.scene_cuts == 1,
          f"{detected.scene_cuts} cut(s) seen, {ignored.scene_cuts} with it disabled")
    check("a cut restarts the noise index",
          detected.frame_index == len(cut_frames) - 2 and ignored.frame_index == len(cut_frames),
          f"index {detected.frame_index} with the detector, {ignored.frame_index} without")
    # The point of the detector is what it is *not* needed for: the learned gate
    # already refuses history that does not match. That is the phase-12 ghosting
    # rejection, on a real scene change rather than a deliberately wrong motion.
    check("the gate rejects the mismatched history on its own",
          ignored_alpha[2] < 0.05,
          f"alpha {ignored_alpha[2]:.4f} across the cut with detection off")

    # the panel's motion multiplier: wrong motion is worse than none, so a scaled-up
    # motion must move the blend away from the correctly-reprojected value
    _, doubled = nr_temporal.run_sequence(model, frames, [m * np.float32(2) for m in motions],
                                          verbose=False)
    check("scaling the motion breaks the reprojection",
          doubled[-1]["alpha_mean"] < settled,
          f"{doubled[-1]['alpha_mean']:.4f} at 2x against {settled:.4f} at 1x")

    print("\n" + ("temporal controls behave" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
