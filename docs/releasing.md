# Releasing SimRig

SimRig releases use one tag-triggered GitHub Actions workflow. The workflow
builds and checks the distributions, publishes them to PyPI with a short-lived
OIDC credential, and creates a GitHub Release with the same artifacts.

## One-time setup

1. Create or sign in to the maintainer account on PyPI and enable two-factor
   authentication.
2. In GitHub, create an environment named `pypi` under **Settings →
   Environments**. Add a deployment approval rule if desired.
3. In PyPI's account publishing settings, add a pending GitHub publisher with:
   - PyPI project name: `simrig`
   - GitHub owner: `Su1eym4n`
   - Repository: `simrig`
   - Workflow: `release.yml`
   - Environment: `pypi`
4. Authenticate the GitHub CLI locally with `gh auth login -h github.com`.

The pending publisher reserves the first trusted release without requiring a
long-lived PyPI API token in GitHub.

## Release checklist

1. Update `simrig/_version.py` and move the release notes out of `Unreleased`
   in `CHANGELOG.md`.
2. Run the test and distribution checks:

   ```bash
   python -m pytest -q
   python -m build
   python -m twine check dist/*
   ```

3. Merge the release commit to `main` and confirm CI is green.
4. Create and push an annotated tag matching the package version exactly:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a v0.2.2 -m "SimRig 0.2.2"
   git push origin v0.2.2
   ```

5. Watch the **Release** workflow. It should publish `simrig==0.2.2` to PyPI
   and create the `v0.2.2` GitHub Release automatically.

If a job fails after the PyPI upload, correct the external configuration and
rerun the failed jobs. Publishing is configured to skip an artifact version
that PyPI already accepted, and the GitHub asset upload is safe to rerun.
