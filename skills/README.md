# SimRig agent skill

Portable skill for Claude Code, Codex, Cursor, and other agents that load
`SKILL.md` packages. Teaches the agent to drive the `simrig` CLI from **any**
working directory (you do not need this repo open).

## Requirements

1. Python 3.10 or newer and the SimRig CLI on PATH:
   ```bash
   # After PyPI publication:
   pip install "simrig[playground]"

   # Before PyPI publication, from a permanent clone:
   git clone https://github.com/Su1eym4n/simrig.git
   cd simrig
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev,playground]"
   python -m pip check
   simrig --help
   ```
2. This skill folder available in the agent’s skills directory (below).

## Install the skill

Codex automatically discovers the repository copy through
`.agents/skills/simrig` when working inside this checkout. No copy is needed for
repository-local use.

For global use from any project, copy the skill from a clone:

```bash
# Codex (all projects)
mkdir -p ~/.agents/skills
cp -R skills/simrig ~/.agents/skills/simrig

# Claude Code (all projects)
mkdir -p ~/.claude/skills
cp -R skills/simrig ~/.claude/skills/simrig

# Cursor
mkdir -p ~/.cursor/skills
cp -R skills/simrig ~/.cursor/skills/simrig
```

When using the pre-PyPI editable installation from another folder, activate the
clone's virtual environment first:

```bash
source /path/to/simrig/.venv/bin/activate
cd /path/to/your/robot-project
simrig --version
```

After the repository is published, Codex can install it from GitHub through
`$skill-installer`:

```text
Use $skill-installer to install the simrig skill from
https://github.com/Su1eym4n/simrig/tree/main/skills/simrig
```

Restart the agent session (Claude Code: check with `/skills`).

## “I don’t see it in Skills”

The **Skills marketplace / curated list** is not the same as **your installed personal skills**.

| App | Where personal skills live | How to see / use |
|-----|----------------------------|------------------|
| Codex | `~/.agents/skills/simrig/` | **New chat**, then type `$simrig` or “use the simrig skill”. |
| Claude Code | `~/.claude/skills/simrig/` | **New session**, run `/skills` or `/simrig`. If `~/.claude/skills` was created mid-session, restart once. |
| Cursor | `~/.agents/skills/simrig/` (and `~/.cursor/skills/simrig/`) | **New Agent chat** — skills load at session start. |

Verify on disk:

```bash
ls ~/.agents/skills/simrig/SKILL.md
ls ~/.claude/skills/simrig/SKILL.md
```

If the file exists but the UI list is empty, start a fresh session and invoke
the skill explicitly (`$simrig` in Codex).

## Use

In any project, ask:

> Use $simrig. Train Go1 joystick walking with a smoke preset.

or

> Use simrig. Here’s my robot XML — scaffold a reach env and smoke-test it.

The agent should run `simrig` commands; it should not re-implement training by hand.
