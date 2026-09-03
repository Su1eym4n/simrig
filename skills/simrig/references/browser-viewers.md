# Browser viewers and custom project UI

Use this reference when choosing a browser viewer, changing its presentation,
or composing a task-specific interface around it.

## Start with a built-in viewer

Choose the viewer that already owns the required lifecycle:

| Need | Viewer |
|---|---|
| Inspect a raw MJCF or Menagerie robot | `simrig view-model` |
| Play a compatible trained checkpoint | `simrig preview` |
| Show an ordinary Python controller that owns MuJoCo stepping | `simrig.LiveWebViewer` |

The built-in pages provide the usual camera interaction and controls. Their
sidebar and Robot View can be collapsed. Prefer that path when it satisfies the
request.

## Compose a custom project interface when useful

A project may need a presentation that the standard viewer does not provide,
such as experiment-specific controls, sensor displays, maps, plots, or other
derived artifacts. In that case, use SimRig for the live simulation scene and
build the surrounding interface as ordinary code in the active project.

Open a browser viewer with `?embed=1` (or `?chrome=0`) to hide SimRig's standard
sidebar, hint, and Robot View while retaining the live viewport. A project page
can embed that URL and arrange its own UI around it. Keep the normal viewer
available unless the user specifically wants only the composed page.

For a running controller, keep `LiveWebViewer` attached to the controller's
existing `MjModel` and `MjData`. Continue sharing `viewer.lock` around MuJoCo
state mutations and call `viewer.sync()` after completed steps. The project may
publish additional endpoints or state for its own interface; choose those from
the actual task rather than assuming a fixed set of panels.

Reuse the embedded viewer instead of copying SimRig's generated Three.js page
into the project. A generally useful improvement discovered while building a
custom interface may be proposed for SimRig, while behavior that still depends
on one robot or experiment can remain with that project.

## Preserve meaning and verify live behavior

- Distinguish a Three.js human-facing view, a native MuJoCo camera render, and
  observations actually consumed by a policy.
- Label scripted, replayed, and learned motion accurately.
- Do not precompute a trajectory merely to make a live controller easier to
  display unless replay is what the user requested.
- Confirm that the embedded scene continues receiving live transforms while
  the custom interface is open.
- Exercise orbit, zoom, and pan in the composed layout, and check its behavior
  at both desktop and narrow widths.
- Verify project-specific displays against the data source that project chose;
  their presence is not evidence that the policy observes or uses that data.
