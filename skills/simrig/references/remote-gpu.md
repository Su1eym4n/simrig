# Remote GPU over SSH

Use this reference for an already-running Linux GPU you can SSH into: a
workstation, a lab box, or a cloud VM. SimRig does not launch or terminate
machines and does not handle cloud API keys.

`--preset large` is PPO scale. `simrig remote` is SSH. Do not conflate them.
Never drop to raw `ssh`/`nohup` when `simrig remote` exists.

## Required sequence

1. Ask the user for the host IP or DNS name and local SSH private-key path if
   they were not supplied. Default username is `ubuntu`.
2. Establish the first connection interactively so the user can verify the
   host fingerprint:

   ```bash
   simrig remote connect HOST --identity KEY
   ```

3. Run `check`, then `prepare`. `prepare` syncs the current SimRig checkout,
   uses system packages in a virtualenv when `--jax-cuda preinstalled`,
   installs `.[playground]`, and requires Python 3.11+ and JAX to report a GPU:

   ```bash
   simrig remote check HOST --identity KEY
   simrig remote prepare HOST --identity KEY
   ```

   Prefer a preinstalled image JAX when it works. If it is absent or
   incompatible, retry intentionally with `--jax-cuda cuda12` or
   `--jax-cuda cuda13` after checking the host driver. Use `--python PATH`
   when the image's default `python3` is older than 3.11.

4. For custom modules, require local static and runtime validation. Then run
   the environment smoke gate on the remote GPU and a foreground PPO smoke run:

   ```bash
   simrig remote smoke HOST ENV_OR_PATH --identity KEY --steps 10
   simrig remote train HOST ENV_OR_PATH --identity KEY --preset smoke
   ```

5. Only after those gates pass and the user explicitly approves the large run:

   ```bash
   simrig remote train HOST ENV_OR_PATH \
     --identity KEY \
     --preset large \
     --detach
   ```

6. Preserve the printed remote output path. Use it for status and fetch:

   ```bash
   simrig remote status HOST REMOTE_OUTPUT --identity KEY --lines 50
   simrig remote fetch HOST REMOTE_OUTPUT --identity KEY
   ```

   After fetch, `simrig status LOCAL_RUN` shows the same pid/progress summary.

7. Evaluate the downloaded policy locally with the exact recorded Python and
   package environment. Treat `--allow-runtime-mismatch` as qualitative only.
   Remind the user to stop a billable VM after artifacts are safe.

Use `--remote-dir` consistently when the user attached a persistent disk.
Never infer that a cloud VM's root disk survives termination.

For browser preview on the remote host, use
`remote connect --tunnel-port 8765`; do not advise opening a public
preview port when an SSH tunnel is sufficient.

When an identity is supplied, require a parseable private key with restrictive
permissions and `IdentitiesOnly=yes`. The first interactive host fingerprint
must be compared with an independent value; `ssh-keyscan` against the same
endpoint is only trust-on-first-use.
