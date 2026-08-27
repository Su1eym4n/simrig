# Remote GPU training over SSH

`simrig remote` uses SSH and `rsync` to run on an **already-running Linux GPU**:
a workstation, a lab box, or a cloud VM you started yourself. SimRig does not
launch or terminate machines, store cloud API keys, or change firewall rules.

`--preset large` is PPO scale. It does not SSH anywhere. A GPU on this machine
uses `simrig train ENV --preset large`. Another machine uses `simrig remote`.

## 1. Have a reachable Linux GPU

The host needs:

- SSH as a normal user (default `ubuntu`; pass `--user` otherwise);
- an NVIDIA GPU visible to `nvidia-smi`;
- Python 3.11+ (`prepare --python` if the default `python3` is older).

Protect the private key:

```bash
chmod 400 ~/.ssh/id_ed25519
```

A cloud VM, a lab box, and a desktop under the desk are the same CLI. If the
image already has JAX+CUDA, keep the default `--jax-cuda preinstalled`.
Otherwise pass `--jax-cuda cuda12` or `cuda13` after checking the driver.

## 2. Establish SSH and check the GPU

Start one interactive connection first so SSH can verify and save the host key:

```bash
simrig remote connect HOST --identity ~/.ssh/id_ed25519
```

Exit that shell, then check NVIDIA and JAX visibility:

```bash
simrig remote check HOST --identity ~/.ssh/id_ed25519
```

The check must list an NVIDIA GPU. Before training, JAX must also report a GPU
device rather than only `CpuDevice`.

When `--identity` is supplied, SimRig validates that the file is parseable,
rejects group/world-readable permissions, and tells OpenSSH to use only that
identity. Compare the first interactive host fingerprint with an independent
value from the machine owner or cloud console; scanning the same endpoint is
only trust-on-first-use.

To reach a remote SimRig preview without opening another firewall port, forward
port 8765 during an interactive session:

```bash
simrig remote connect HOST \
  --identity ~/.ssh/id_ed25519 \
  --tunnel-port 8765
```

Then a preview listening on remote `127.0.0.1:8765` is available locally at
`http://127.0.0.1:8765/`.

## 3. Sync and prepare SimRig

From the SimRig source checkout containing the robot, scene, and custom env:

```bash
simrig remote prepare HOST \
  --identity ~/.ssh/id_ed25519 \
  --python python3.12
```

`prepare`:

- syncs the checkout to `/home/<user>/simrig` with `rsync`;
- excludes Git metadata, local virtualenvs, caches, reports, and existing runs;
- requires Python 3.11 or newer so pip cannot silently select the older
  Playground 0.1 stack;
- recreates `.venv` cleanly, with the host's system packages visible only in
  `--jax-cuda preinstalled` mode;
- installs the checkout with the `playground` extra;
- stops with an error unless JAX can see a GPU.

Pass `--remote-dir /path/on/host/simrig` to keep the checkout and runs on a
persistent disk (for example an attached cloud filesystem). Use the same
`--remote-dir` on subsequent commands. Never infer that a cloud VM's root disk
survives termination.

## 4. Run the safety gates

For a custom environment, finish local static and runtime validation before
syncing:

```bash
simrig validate-env envs/my_task.py
simrig validate-env envs/my_task.py --runtime
```

Run reset/step smoke on the remote GPU:

```bash
simrig remote smoke HOST envs/my_task.py \
  --identity ~/.ssh/id_ed25519 \
  --steps 10
```

Then verify the full PPO path with the small training preset. Keep this run in
the foreground so failures are immediately visible:

```bash
simrig remote train HOST envs/my_task.py \
  --identity ~/.ssh/id_ed25519 \
  --preset smoke \
  --impl auto \
  --seed 0
```

Compilation, device visibility, and smoke success do not prove the requested
behavior has been learned. Inspect the metrics and evaluate the smoke
checkpoint before spending on a larger run.

If the user wants several behaviors (walk and jump, reach then hold), smoke
and train each primitive env first. Do not start a detached `--preset large`
run on a composed task until those primitives work, or the user explicitly
wants one policy.

## 5. Start and monitor a large run

After every earlier gate passes, start the large preset in detached mode:

```bash
simrig remote train HOST envs/my_task.py \
  --identity ~/.ssh/id_ed25519 \
  --preset large \
  --impl auto \
  --seed 0 \
  --detach
```

The command prints the exact remote output directory. Save it. Detached runs
write `train.log`, `train.pid`, `metrics.jsonl`, and `progress.json` inside
that directory. Check progress with:

```bash
simrig remote status HOST REMOTE_OUTPUT \
  --identity ~/.ssh/id_ed25519 \
  --lines 50
```

After fetching, the same summary is local:

```bash
simrig status runs/RUN
```

Resume a crashed or shorter run from its checkpoints:

```bash
simrig remote train HOST envs/my_task.py \
  --identity ~/.ssh/id_ed25519 \
  --resume REMOTE_OUTPUT \
  --preset large \
  --detach
```

`--resume` is a path **on the remote host**. Locally, use
`simrig train ENV --resume runs/RUN`.

You can replace the preset scale deliberately with `--timesteps`, `--num-envs`,
or `--batch-size`. Large values can exhaust GPU memory; start from the tested
smoke configuration and change one dimension at a time.

## 6. Download and evaluate the result

Copy the complete run directory back to local `runs/`:

```bash
simrig remote fetch HOST REMOTE_OUTPUT \
  --identity ~/.ssh/id_ed25519
```

Evaluate and preview the fetched checkpoint with the exact same environment:

```bash
simrig eval runs/RUN/policy.params \
  --env envs/my_task.py \
  --steps 500 \
  --seed 0

simrig preview runs/RUN/policy.params \
  --env envs/my_task.py \
  --port 8765
```

If the environment exposes `state.metrics["success"]` or a `SUCCESS_SPEC`,
eval reports `task_success` as a boolean. Otherwise it stays unknown; rollout
completion is not success.

After confirming that artifacts are local or on persistent storage, stop a
billable VM in its cloud console. SimRig does not stop it.

## Troubleshooting

| Symptom | First check |
|---|---|
| SSH asks about an unknown host | Run `remote connect` interactively once and verify the fingerprint |
| Invalid or rejected identity file | Confirm the private key matches the host, is a real OpenSSH/PEM key, and has permissions `400` or `600` |
| Remote Python is too old | Select Python 3.11+ with `prepare --python /path/to/python` |
| `rsync` is missing locally | Install `rsync`; both `prepare` and `fetch` require it |
| `nvidia-smi` fails | Confirm the host has an NVIDIA GPU and working drivers |
| JAX reports only CPU | Recreate the venv with `prepare`; inspect JAX/CUDA vs the driver |
| PPO runs out of memory | Reduce `--num-envs` and `--batch-size` |
| Detached run says `stopped` | Inspect `train.log`, fix the earliest error, and rerun the failed gate |
