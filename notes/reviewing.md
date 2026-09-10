# Reviewing this repository

`/ultrareview` needs a base branch and a diff under its size limit. This repository has
a single line of history on `master`, so a first review of the whole tree is **180 files
and 25 903 insertions** and is refused. Four base branches cut it into areas: each one
is `master` with one area removed, so `master` against it shows exactly that area as
additions.

```sh
git checkout review-layer && /ultrareview review-base && git checkout master
```

and the same for `review-gpu`, `review-ref`, `review-tools`.

| branch | files | lines |
|---|---|---|
| `review-layer` | 11 | 1 362 — start here |
| `review-gpu` | 29 | 5 404 |
| `review-ref` | 15 | 4 005 |
| `review-tools` | 43 | 3 068 |

Each is an **orphan** branch with exactly one commit on top of `review-base`, and that
commit's patch is the area. `review-base` carries `CLAUDE.md`, `HANDOFF.md`, this file
and the `Makefile`, so a reviewer can see what the project is without any of it counting
as changes.

**Why orphans, which is not obvious.** The first attempt built the bases as *descendants*
of master — take master, delete the area, commit — reasoning that `master` against such a
branch is a tree diff showing the area as additions. `git diff` agrees. The review tool
does not: it reviews the **patches of the commits unique to the branch**, and the only
commit master had that the base lacked was the one adding this file. The reported scope
came out as "1 file changed, 65 insertions" and a review was spent on it. If the scope
line does not match the table above, the branches are wrong again.

Always pass a base: bare `/ultrareview` looks for `main`, which does not exist, and
creating one would silently make some arbitrary scope the default.

## What is worth a reviewer's time, in order

**`base-layer` — `src/layer`, and the reason to start here.** It is C that loads
*inside another process*, in this case a game running under Wine. It intercepts
`vkQueuePresentKHR`, copies swapchain images, talks to a daemon over a Unix socket, and
hands the result back. A bug here does not produce a wrong picture; it takes the host
game down. It has already had one such bug fixed — `write()` instead of
`send(..., MSG_NOSIGNAL)`, where a daemon dying mid-transfer would kill the game with
SIGPIPE. Look at lifetimes, partial reads and writes, the two ABIs, and what happens
when the daemon is absent, slow, or lying about sizes.

**`base-gpu` — `src/gpu`.** The resident runtime: `libxmx.c` owns the Vulkan device,
buffers and pipelines and records whole frames; the `.comp` shaders are the graph. The
sharpest things to check are the ones this project got wrong before: buffer sizing and
offsets around `xmx_rec_gemm` (cooperative-matrix loads are **not** bounds-checked here,
`cooperativeMatrixRobustBufferAccess` is false, so an overrun is silent), the scratch
arena's aliasing plan against the barriers in `nr_resident.py`, and whether every
`coopMatStore` writes into an array of the matrix's own component type — the one rule
whose violation cost this project three phases (`notes/phase38-there-was-no-bug.md`).

**`review-ref` — `src/ref`.** The CPU reference and the frame/temporal drivers. The live
files are `nr_model.py`, `nr_frame.py`, `nr_temporal.py`, `nr_display.py`, `nr_accel.py`,
`image_io.py` and their tests.

`hnet_model.py`, `hnet_ops.py`, `hnet_ref.py`, `forward.py` and `run_frame.py` are
**superseded** — they decode the weight container the wrong way — and are kept for the
PTX-derived findings they encode. They are not worth reviewing *as production code*, but
"do not read them" was too strong, and a review said so: **two tests in `src/gpu` import
from them** — `test_attention_gpu.py` takes `Model`, `softmax`, `l2_normalize`,
`HEAD_DIM`, `TOKENS` and `GQA_RATIO`, and `test_layer.py` takes `Model`. Whoever reviews
`review-gpu` will meet those imports and needs to know what is behind them.

### The two tests outside `make test`, and why

Both load `work/weights_ht.bin`, which is carved out of the DLL and gitignored, so
neither can run on a fresh clone — that alone keeps them out of the recipe. Beyond that
they differ, and the difference matters:

- `test_attention_gpu.py` **passes**: worst relative deviation **2.8e-04**, Phase 4's
  acceptance result. It asks whether the GPU path reproduces the CPU path *on the same
  weights*, so a wrong decode does not invalidate it. It is a kernel test.
- `test_layer.py` **fails at 0.22 and cannot pass.** The dense-FP16 decode yields values
  including FP16 subnormals; XMX flushes subnormal operands to zero and the float64
  reference does not. The premise it was written under — "27 % of this model's parameters
  are subnormal" — was an artefact of the same wrong decode, and the real figure is
  0.00006 %.

Both now say this at the top of the file, so nobody runs them expecting green.

**`base-tools` — `src/bench`, `src/probe`, `src/tools`.** Measurement and
reverse-engineering tooling. Lowest risk, but the benchmarks are where claims come from,
so an error here is an error in the notes. Worth checking that each one measures what
its name says: several findings in `notes/` turned on a benchmark's warm-up, its pairing,
or its expectations being wrong rather than the code under test.

## What not to review

`notes/` is 7 239 lines of prose and is the project's record, including its wrong turns,
which are marked as such. It is not code and it will consume a reviewer for nothing.

## Ground truth for "is this still correct"

```sh
make test          # 95 checks, including bit-exactness against the CPU reference
```

The number that matters is the resident graph's correlation with the host reference,
**0.981311**. It has not moved through any optimisation in this repository, and a change
that moves it is either a bug or a deliberate, documented trade.
