# Tasks

## 1. Run the full study

The grid is 6 arms x 5 pool sizes x 3 seeds, so 90 cells. The 1M and 2.46M rows read far
more data per step than the smaller ones, so budget the wall clock from those.

```bash
# local
make setup
make test
cp server.conf.example server.conf     # edit HOST, PORT, KEY, REMOTE, GPUS
make push
```

```bash
# on the server
uv sync --group gpu --extra pizza
make prepare                       # clone PIZZA, compute max_new_tokens per arm
make check                         # stop here if it fails
make smoke                         # n=200, 100 steps, under 10 minutes

./scripts/tmux.sh sweep 3          # LR sweep at n=100k
./scripts/tmux.sh select           # pick LRs once every shard is done
./scripts/tmux.sh grid 3           # the 90 cells
./scripts/tmux.sh latency 3        # sharded, not a single session
./scripts/tmux.sh single determinism test grammar fewshot report

./scripts/tmux.sh status           # progress
./scripts/tmux.sh tail             # follow every shard log
```

## 2. Rebuild the paper from the new runs

```bash
make fetch            # pull runs/ back, adapters skipped
make status
make paper            # figures, tables and paper.pdf
make -C paper check   # page count, overfull boxes, em dash count
```

`paper/generated/` and `paper/figures/` regenerate themselves. Then re-copy the results
table in `README.md`, which is hand written from the macros, and check any number in
`paper.tex` prose that is not wrapped in a macro.

## 3. Re-measure on device

Only needed when the model changes.

1. copy `coffee-q8_0.gguf` into the app: Finder -> iPad -> Files -> Loop
2. build `../loop-coffee-app` to the iPad, not a simulator
3. triple-tap the wordmark, then **Бенчмарк** in the Orders toolbar, **Запустить**
4. **Копировать**, and paste the `\renewcommand{\pad*}{...}` block over the macros at the
   top of `paper/paper.tex`

## 4. Full-data comparison with the published baseline

The protocol declines a state-of-the-art claim because the published BART-Large EXR result
(78.56 unordered EM) used the full 2.46M training set. Now that 2.46M is a pool size, run
the EXR arm at full data and report it as a like-for-like comparison, or state why the
architectures still are not comparable.

## 5. Replace the café eval set

The 49 case eval set and the training generator were written by the same person, so 98
percent likely overstates real performance. Collect real speech-to-text transcripts from
the booth and re-measure against those.
