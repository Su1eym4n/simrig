# Lambda Cloud GPU training

This page moved. SimRig talks to **any already-running Linux GPU over SSH**,
not only Lambda.

See [Remote GPU training over SSH](remote-gpu.md).

The old command `simrig cloud lambda …` was removed. Use `simrig remote …`.
`--preset cloud` is a hidden alias for `--preset large` (PPO scale, not SSH).
