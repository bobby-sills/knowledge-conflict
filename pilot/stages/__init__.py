"""Stage entry points. Each is idempotent, resumable, and writes to `results/`.

Run order:

    00  factset      build the fact set, assign splits            (CPU)
    01  prior        closed-book screening -> conflict states      (GPU)
    02  capture      the one expensive pass: lens + all signals    (GPU)
    03  test1        internal vs external knowledge                (CPU)
    04  test2        the resist-or-correct AUC gate                (CPU)
    05  test3a       analytic reachability, tau*                    (CPU)
    06  test3b       tau sweep, baselines, oracle routing          (GPU)
    07  test4        permutation control                           (GPU for the
                                                                   decoding half)
    08  timing       two-pass overhead measurement                 (GPU)

Stages 03-05 and the cheap half of 07 are pure re-analyses of stage 02's output,
which is the point: a dead session costs nothing after capture is complete.
"""
