# Lambda Cloud GPU training

SimRig can use SSH and `rsync` to run on an already-provisioned Lambda
On-Demand Cloud GPU. It does not launch or terminate billable instances, store
Lambda API keys, or change firewall rules.

## 1. Launch the instance

In the [Lambda Cloud console](https://cloud.lambda.ai/), add or generate an SSH
key, then launch an On-Demand instance with that key. The Lambda Stack image is
the easiest starting point because it includes NVIDIA drivers, CUDA, JAX, and
other ML packages. An attached Lambda filesystem is optional, but useful when
runs must survive instance termination.

Copy the public IP from the Instances page. If Lambda generated the private
key, protect the downloaded file locally:

```bash
chmod 400 ~/Downloads/lambda-key.pem
```

Lambda's current instance and SSH instructions are in its
[Cloud console](https://docs.lambda.ai/public-cloud/console/) and
[Connecting to an instance](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)
documentation.

## 2. Establish SSH and check the GPU

Start one interactive connection first. This lets standard SSH verify and save
the host key before later non-interactive commands use batch mode:

```bash
simrig cloud lambda connect INSTANCE_IP \
  --identity ~/Downloads/lambda-key.pem
```

Exit that shell, then check NVIDIA and JAX visibility:

```bash
simrig cloud lambda check INSTANCE_IP \
  --identity ~/Downloads/lambda-key.pem
```

The check must list an NVIDIA GPU. Before training, JAX must also report a GPU
device rather than only `CpuDevice`.

When `--identity` is supplied, SimRig validates that the file is parseable,
rejects group/world-readable permissions, and tells OpenSSH to use only that
identity. Compare the first interactive host fingerprint with the fingerprint
shown in the Lambda console; scanning and trusting the same network endpoint is
only trust-on-first-use.

To reach a remote SimRig preview without opening another firewall port, forward
port 8765 during an interactive session:

```bash
simrig cloud lambda connect INSTANCE_IP \
  --identity ~/Downloads/lambda-key.pem \
  --tunnel-port 8765
```

Then a preview listening on remote `127.0.0.1:8765` is available locally at
`http://127.0.0.1:8765/`.

## 3. Sync and prepare SimRig

From the SimRig source checkout containing the robot, scene, and custom env:

```bash
simrig cloud lambda prepare INSTANCE_IP \
  --identity ~/Downloads/lambda-key.pem \
  --python python3.12
```

`prepare`:

- syncs the checkout to `/home/ubuntu/simrig` with `rsync`;
- excludes Git metadata, local virtualenvs, caches, reports, and existing runs;
- requires Python 3.11 or newer so pip cannot silently select the older
  Playground 0.1 stack;
- recreates `.venv` cleanly, with Lambda's preinstalled system packages visible
  only in `--jax-cuda preinstalled` mode;
- installs the checkout with the `playground` extra;
- stops with an error unless JAX can see a GPU.

The Playground extra pins the validated SimRig training stack. Use `--python`
when the image's default `python3` is older than 3.11. Select an image with a
suitable Python or install it through the image's normal environment-management
workflow before running `prepare`; SimRig does not modify the system Python.

The default `--jax-cuda preinstalled` mode follows Lambda's recommendation to
reuse Lambda Stack packages. If the chosen image does not provide a usable JAX,
use `--jax-cuda cuda12` or `--jax-cuda cuda13` to install JAX's pip-bundled CUDA
runtime instead. Match it to the installed NVIDIA driver; CUDA 12 is the more
compatible fallback for older drivers. See JAX's current
[NVIDIA GPU installation matrix](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

Pass `--remote-dir /lambda/nfs/FILESYSTEM_NAME/simrig` to keep the checkout and
runs on an attached persistent Lambda filesystem. Use the same `--remote-dir`
on subsequent commands.

## 4. Run the safety gates

For a custom environment, finish local static and runtime validation before
syncing:

```bash
simrig validate-env envs/my_task.py
simrig validate-env envs/my_task.py --runtime
```

Run reset/step smoke on the Lambda GPU:

```bash
simrig cloud lambda smoke INSTANCE_IP envs/my_task.py \
  --identity ~/Downloads/lambda-key.pem \
  --steps 10
```

Then verify the full PPO path with the small training preset. Keep this run in
the foreground so failures are immediately visible:

```bash
simrig cloud lambda train INSTANCE_IP envs/my_task.py \
  --identity ~/Downloads/lambda-key.pem \
  --preset smoke
```

Compilation, device visibility, and smoke success do not prove the requested
behavior has been learned. Inspect the metrics and evaluate the smoke
checkpoint before spending on a larger run.

## 5. Start and monitor a cloud run

After every earlier gate passes, start the large preset in detached mode:

```bash
simrig cloud lambda train INSTANCE_IP envs/my_task.py \
  --identity ~/Downloads/lambda-key.pem \
  --preset cloud \
  --detach
```

The command prints the exact remote output directory. Save it. Detached runs
write `train.log` and `train.pid` inside that directory. Check progress with:

```bash
simrig cloud lambda status INSTANCE_IP REMOTE_OUTPUT \
  --identity ~/Downloads/lambda-key.pem \
  --lines 50
```

You can replace the preset scale deliberately with `--timesteps`, `--num-envs`,
or `--batch-size`. Large values can exhaust GPU memory; start from the tested
smoke configuration and change one dimension at a time.

## 6. Download and evaluate the result

Copy the complete run directory back to local `runs/`:

```bash
simrig cloud lambda fetch INSTANCE_IP REMOTE_OUTPUT \
  --identity ~/Downloads/lambda-key.pem
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

Training records Python and package versions in `config.json`. Eval, demo, and
preview refuse a different recorded runtime by default. If exact recreation is
impossible, `--allow-runtime-mismatch` permits an explicitly qualitative check;
it is not equivalent evaluation evidence.

The generic evaluator reports rollout completion, termination, and reward, but
sets task success to unknown. Command tracking, foot slip, energy, or other
task-specific outcomes require a separate evaluator and explicit pass criteria.

After confirming that artifacts are local or on persistent storage, terminate
the instance from the Lambda console to stop compute charges. SimRig does not
terminate it automatically.

## Troubleshooting

| Symptom | First check |
|---|---|
| SSH asks about an unknown host | Run `cloud lambda connect` interactively once and verify the fingerprint |
| Invalid or rejected identity file | Confirm the private key matches the key selected at launch, is a real OpenSSH/PEM key rather than a path stored inside a text file, and has permissions `400` or `600` |
| Remote Python is too old | Select Python 3.11+ with `prepare --python /path/to/python` |
| `rsync` is missing locally | Install `rsync`; both `prepare` and `fetch` require it |
| `nvidia-smi` fails | Confirm the instance is a GPU type and uses a Lambda Stack or GPU Base image |
| JAX reports only CPU | Recreate the venv with `prepare`; inspect the base image and JAX/CUDA compatibility |
| PPO runs out of memory | Reduce `--num-envs` and `--batch-size` |
| Detached run says `stopped` | Inspect `train.log`, fix the earliest error, and rerun the failed gate |

Lambda documents its current base images and preinstalled software in
[On-Demand Cloud overview](https://docs.lambda.ai/public-cloud/on-demand/) and
[Managing your system environment](https://docs.lambda.ai/public-cloud/on-demand/managing-system-environment/).
