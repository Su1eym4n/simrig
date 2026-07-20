# SimRig agent skill

Portable skill for Claude Code, Codex, Cursor, and other agents that load
`SKILL.md` packages. Teaches the agent to drive the `simrig` CLI from **any**
working directory (you do not need this repo open).

## Requirements

1. SimRig CLI on PATH:
   ```bash
   pip install "simrig[playground]"
   # until published to PyPI, from a clone:
   pip install -e ".[playground]"
   simrig --help
   ```
2. This skill folder installed into the agent’s skills directory (below).

## Install the skill

From a clone of this repo:

```bash
# Claude Code (all projects)
mkdir -p ~/.claude/skills
cp -R skills/simrig ~/.claude/skills/simrig

# Codex
mkdir -p ~/.codex/skills
cp -R skills/simrig ~/.codex/skills/simrig

# Cursor
mkdir -p ~/.cursor/skills
cp -R skills/simrig ~/.cursor/skills/simrig
```

Or with the skills CLI when published:

```bash
npx skills add <github-user>/text-to-train --skill simrig
```

Restart the agent session (Claude Code: check with `/skills`).

## “I don’t see it in Skills”

The **Skills marketplace / curated list** is not the same as **your installed personal skills**.

| App | Where personal skills live | How to see / use |
|-----|----------------------------|------------------|
| Codex | `~/.codex/skills/simrig/` | **New chat**, then type `/simrig` or “use the simrig skill”. Marketplace UI only shows curated OpenAI skills. |
| Claude Code | `~/.claude/skills/simrig/` | **New session**, run `/skills` or `/simrig`. If `~/.claude/skills` was created mid-session, restart once. |
| Cursor | `~/.agents/skills/simrig/` (and `~/.cursor/skills/simrig/`) | **New Agent chat** — skills load at session start. |

Verify on disk:

```bash
ls ~/.codex/skills/simrig/SKILL.md
ls ~/.claude/skills/simrig/SKILL.md
```

If the file exists but the UI list is empty, the install worked — invoke it with `/simrig` in a fresh session.

## Use

In any project, ask:

> Use the simrig skill. Train Go1 joystick walking with a smoke preset.

or

> Use simrig. Here’s my robot XML — scaffold a reach env and smoke-test it.

The agent should run `simrig` commands; it should not re-implement training by hand.
