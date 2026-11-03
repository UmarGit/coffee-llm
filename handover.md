# Handover: running the training on the GPU server

For someone running this for the first time. Code is edited on the user, sent
to the server, trained there, and the results are pulled back to the user.
The server has no GitHub access and no rsync, so everything is sent from the
user over ssh by the scripts below.

`[user]` means a terminal in this repo on the user's computer. `[server]` means a terminal
on the GPU server, in `~/experiments`.

## Part 1: one-time setup

### 1.1 user: tools and the repo

```bash
[user]  uv --version                    # install uv first if missing: https://docs.astral.sh/uv/
[user]  uv sync --extra pizza           # local environment (no GPU packages)
[user]  uv run pytest -q tests          # expect: 81 passed, 3 skipped
```

The PIZZA dataset must be on the user in `data/external/pizza/` (it is
git-ignored). If it is missing:

```bash
[user]  git clone --depth 1 https://github.com/amazon-science/pizza-semantic-parsing-dataset.git data/external/pizza
```

### 1.2 user: server.conf (the connection settings)

Every script reads the server address from `server.conf` in the repo root.
It is git-ignored, so each person makes their own:

```bash
[user]  cp server.conf.example server.conf
```

Edit `server.conf`:

| Setting | Meaning | Current value |
|---|---|---|
| `HOST` | user@address of the GPU server | `user-8@10.32.15.90` |
| `PORT` | ssh port | `2235` |
| `KEY` | your private ssh key for that server | `$HOME/.ssh/custodian_ed25519` |
| `REMOTE` | repo folder on the server, relative to the server's home | `experiments` |
| `GPUS` | GPUs to use; `auto` = all | `auto` |

The server owner must have added your public key (`KEY` + `.pub`) to the
server. Check the connection and create the repo folder:

```bash
[user]  source server.conf
[user]  ssh -i "$KEY" -p "$PORT" "$HOST" "mkdir -p ~/$REMOTE && nvidia-smi -L"   # lists 3 GPUs
```

To log in to the server later:

```bash
[user]  source server.conf && ssh -i "$KEY" -p "$PORT" "$HOST"
```

### 1.3 Send the dataset (once)

About 165 MB compressed:

```bash
[user]  source server.conf
[user]  tar czf - --exclude .git -C data/external pizza \
         | ssh -i "$KEY" -p "$PORT" "$HOST" "mkdir -p ~/$REMOTE/data/external && tar xzf - -C ~/$REMOTE/data/external"
```

### 1.4 Send llama.cpp (once, needed to export the GGUF model)

```bash
[user]  ./scripts/push_llama_cpp.sh       # lands in ~/llama.cpp on the server; built there on first export
```

### 1.5 Send the code and install the environment

```bash
[user]     make push                       # sends src/, scripts/, tests/, Makefile, ...; never runs/ or data/
[server]  cd ~/experiments
[server]  uv --version                    # if missing: pip install --user uv
[server]  uv sync --extra pizza --group gpu
[server]  ls data/external/pizza/utils/catalogs   # the dataset arrived
```

## Part 2: every time you change code

```bash
[user]     uv run pytest -q tests
[user]     make push
[server]  cd ~/experiments && cat VERSION         # the commit you just sent ("-dirty" = uncommitted changes)
[server]  uv sync --extra pizza --group gpu       # only if pyproject.toml / uv.lock changed
```

`make push` never deletes or overwrites results on the server.

## Part 3: the runs (all fresh)

All long jobs run in tmux sessions, so they keep running if ssh drops.

```bash
[server]  ./scripts/tmux.sh status     # what is running, campaign progress
[server]  ./scripts/tmux.sh stop       # stop everything
```

### 3.1 Prepare (both models)

Computes the token caps for each model (`runs/qwen/pizza/caps.json`,
`runs/lfm2/pizza/caps.json`). Needed before the candidate models; the
campaigns do it themselves.

```bash
[server]  for m in qwen lfm2; do
            uv run --extra pizza --group gpu python -m bench prepare --model $m --task pizza
            uv run --extra pizza --group gpu python -m bench check --model $m --task pizza --negatives 0.3 --corrections 0.1
          done                                                   # each check ends "all checks passed"
```

### 3.2 Candidate models (WORDS, negatives 30%, corrections 10%)

One Qwen and one LFM2 model, side by side on GPUs 0 and 1. About 20 minutes
each: trains, converts to GGUF, verifies the GGUF. Learning rate 0.0001 is
what the paper's sweep chose for WORDS on both models.

```bash
[server]  GPU=0 ./scripts/tmux.sh export qwen words pizza --fresh --negatives 0.3 --corrections 0.1 --lr 0.0001
[server]  GPU=1 ./scripts/tmux.sh export lfm2 words pizza --fresh --negatives 0.3 --corrections 0.1 --lr 0.0001
[server]  tail -f runs/qwen/pizza/logs/export-words.log     # ctrl-c stops following, not the job
[server]  tail -f runs/lfm2/pizza/logs/export-words.log
```

The end of each log prints the `negatives` and `corrections` results. They
are also in `runs/<model>/pizza/words-n2456443-s0-lr0.0001-neg30-cor10-fresh-<date>/metrics.json`,
and the GGUFs with their manifests in `models/pizza/`.

### 3.3 Pilot campaign (both models, tiny, under an hour)

Proves the whole pipeline works before the big run. Results go to
`runs-neg-v1-pilot/`, never `runs/`.

```bash
[server]  ./scripts/tmux.sh campaign neg-v1 --pilot --negatives 0.3 --corrections 0.1
[server]  tail -f runs/logs/campaign-neg-v1.log
[server]  cat runs-neg-v1-pilot/bundle/summary.txt          # must end with ALL CHECKS PASSED
```

A campaign uses every GPU on the machine. To use only some cards, put
`GPU_IDS` in front of the command: `GPU_IDS=0,2 ./scripts/tmux.sh campaign ...`.
The pilot and the full campaign share one tmux session name, so close the
finished pilot first: `tmux kill-session -t bench-campaign-neg-v1`.

### 3.4 Full campaign (after the researchers approve the setup)

Both models, every stage, a GGUF of WORDS for each model and a results bundle, in
`runs-neg-v1/`. Takes days.

```bash
[server]  ./scripts/tmux.sh campaign neg-v1 --negatives 0.3 --corrections 0.1
[server]  cat runs-neg-v1/bundle/summary.txt                # when finished
```

Notes:

- All six formats train on the same negatives and corrections. Each refuses
  in its own spelling: `?` (words, symbols, tags), `(REJECT )` (brackets),
  `(R)` (short brackets), `{"reject":true}` (JSON).
- A crashed or stopped campaign resumes when you run the same command again.
- A campaign name is tied to its flags. To change the flags, use a new name.
- The default is 3 seeds. For more, add `--seeds 0,1,2,3,4` from the first start.

## Part 4: pull the results to the user

```bash
[user]  ./scripts/fetch_model.sh qwen-words-n2456443-s0-lr0.0001-neg30-cor10   # Qwen candidate (resumable, sha256 checked)
[user]  ./scripts/fetch_model.sh lfm2-words-n2456443-s0-lr0.0001-neg30-cor10   # LFM2 candidate
[user]  ./scripts/fetch_runs.sh                           # runs/ without adapters -> runs/
[user]  ./scripts/fetch_campaign.sh neg-v1-pilot          # pilot bundle -> runs-neg-v1-pilot/
[user]  ./scripts/fetch_campaign.sh neg-v1 --models       # full campaign bundle and both GGUFs -> runs-neg-v1/
```

The models land in `site/public/models/`: Qwen as `pizza-words-q8_0.gguf`
(replacing the one the local site dev server uses; the live website is not
touched), LFM2 as `pizza-lfm2-words-q8_0.gguf`.

Never commit anything pulled from the server: the PIZZA dataset is
CC-BY-NC-4.0, and `runs*/`, `models/` and `*.gguf` are git-ignored for that
reason.
