# Install the SimRig agent skill

This page is for people installing or troubleshooting the portable SimRig
agent skill. For normal SimRig setup and use, start with the repository
[README](../README.md). For development and tests, use
[CONTRIBUTING.md](../CONTRIBUTING.md).

The Python package and the agent skill are separate:

- The `simrig` CLI runs inspection, training, evaluation, and previews.
- The agent skill teaches Codex, Claude Code, or Cursor how to use that CLI.

Install the CLI before installing the skill. Until SimRig is on PyPI, follow the
GitHub clone and editable-install instructions in the main README.

## Repository-local use

Codex discovers `.agents/skills/simrig` automatically while working inside this
repository. Contributors do not need to copy the skill globally.

## Global use from any project

The preferred installation uses the Skills CLI:

```bash
npx skills add Su1eym4n/simrig --skill simrig --global
```

Target one agent non-interactively when needed:

```bash
npx skills add Su1eym4n/simrig \
  --skill simrig \
  --global \
  --agent codex \
  --yes
```

When using the pre-PyPI editable installation, activate the clone's virtual
environment before launching the agent or running SimRig:

```bash
source /path/to/simrig/.venv/bin/activate
cd /path/to/your/robot-project
simrig --version
```

Start a new agent task, then invoke `$simrig` explicitly or ask naturally for a
MuJoCo training workflow.

## Codex skill installer

Codex can install the skill through `$skill-installer`:

```text
Use $skill-installer to install the simrig skill from
https://github.com/Su1eym4n/simrig/tree/main/skills/simrig
```

## Manual installation

Run the relevant command from a SimRig clone:

```bash
# Codex
mkdir -p ~/.agents/skills
cp -R skills/simrig ~/.agents/skills/simrig

# Claude Code
mkdir -p ~/.claude/skills
cp -R skills/simrig ~/.claude/skills/simrig

# Cursor
mkdir -p ~/.cursor/skills
cp -R skills/simrig ~/.cursor/skills/simrig
```

## Troubleshooting

Personal skill locations:

| App | Skill location | Invocation |
|---|---|---|
| Codex | `~/.agents/skills/simrig/` | Start a new task, then type `$simrig`. |
| Claude Code | `~/.claude/skills/simrig/` | Start a new session, then use `/skills` or `/simrig`. |
| Cursor | `~/.cursor/skills/simrig/` | Start a new Agent task. |

Verify the installation:

```bash
ls ~/.agents/skills/simrig/SKILL.md
simrig --version
```

If the files exist but the skill is not shown, restart the agent and invoke it
explicitly. Avoid keeping repository and global copies active with the same
name while developing SimRig, because both can appear in skill selectors.
