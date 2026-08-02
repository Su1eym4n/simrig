# Lambda Cloud operations

Use this reference only for an already-provisioned Lambda On-Demand Cloud GPU.
SimRig's SSH workflow intentionally does not launch or terminate billable
instances and does not handle Lambda API keys.

## Required sequence

1. Ask the user for the instance IP and local SSH private-key path if they were
   not supplied. The default Lambda username is `ubuntu`.
2. Establish the first connection interactively so the user can verify the
   host fingerprint:

   ```bash
   simrig cloud lambda connect HOST --identity KEY
   ```

3. Run `check`, then `prepare`. `prepare` syncs the current SimRig checkout,
   uses Lambda's system packages in a virtualenv, installs `.[playground]`, and
   requires Python 3.11+ and JAX to report a GPU:

   ```bash
   simrig cloud lambda check HOST --identity KEY
   simrig cloud lambda prepare HOST --identity KEY
   ```

   Prefer the default preinstalled Lambda Stack JAX. If it is absent or
   incompatible, retry intentionally with `--jax-cuda cuda12` or
   `--jax-cuda cuda13` after checking the instance driver. Use `--python PATH`
   when the image's default `python3` is older than 3.11. Do not accept a pip
   backtrack to an older Playground release as a compatibility fix.

4. For custom modules, require local static and runtime validation. Then run
   the environment smoke gate on Lambda and a foreground PPO smoke run:

   ```bash
   simrig cloud lambda smoke HOST ENV_OR_PATH --identity KEY --steps 10
   simrig cloud lambda train HOST ENV_OR_PATH --identity KEY --preset smoke
   ```

5. Only after those gates pass and the user explicitly approves the large run:

   ```bash
   simrig cloud lambda train HOST ENV_OR_PATH \
     --identity KEY \
     --preset cloud \
     --detach
   ```

6. Preserve the printed remote output path. Use it for status and fetch:

   ```bash
   simrig cloud lambda status HOST REMOTE_OUTPUT --identity KEY --lines 50
   simrig cloud lambda fetch HOST REMOTE_OUTPUT --identity KEY
   ```

7. Evaluate the downloaded policy locally with the exact recorded Python and
   package environment. Treat `--allow-runtime-mismatch` as qualitative only.
   Remind
   the user to terminate the Lambda instance after artifacts are downloaded or
   confirmed on an attached persistent filesystem.

Use `--remote-dir /lambda/nfs/FILESYSTEM_NAME/simrig` consistently when the
user attached a Lambda filesystem. Never infer that the root disk survives
instance termination.

For browser preview on the remote instance, use
`cloud lambda connect --tunnel-port 8765`; do not advise opening a public
preview port when an SSH tunnel is sufficient.

When an identity is supplied, require a parseable private key with restrictive
permissions and `IdentitiesOnly=yes`. The first interactive host fingerprint
must be compared with an independent value from the cloud console; `ssh-keyscan`
against the same endpoint is only trust-on-first-use.
