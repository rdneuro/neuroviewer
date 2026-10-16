# Releasing

## One-time setup

1. **Create the GitHub repository** `rdneuro/neuroviewer` and push `main`.
2. **Reserve the PyPI name with trusted publishing** (no API token needed):
   on <https://pypi.org/manage/account/publishing/>, add a *pending
   publisher* with
   - PyPI project name: `neuroviewer`
   - Owner: `rdneuro`, Repository: `neuroviewer`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On GitHub, *Settings > Environments*, create an environment called `pypi`
   (optionally require your approval before each upload).

## Every release

1. Bump `__version__` in `src/neuroviewer/__init__.py`.
2. Add the section to `CHANGELOG.md`.
3. Check locally:
   ```bash
   pip install -e ".[dev]"
   pytest
   python -m build && twine check --strict dist/*
   ```
4. Commit, tag and push:
   ```bash
   git commit -am "Release 0.1.1"
   git tag v0.1.1
   git push origin main --tags
   ```
5. The `Release` workflow checks that the tag matches `__version__`,
   publishes the wheel and sdist to PyPI and attaches
   `Neuroviewer-<version>-x86_64.AppImage` to the GitHub release.

## Manual upload (fallback)

```bash
python -m build
twine upload --repository testpypi dist/*   # optional dry run on TestPyPI
twine upload dist/*
```

PyPI never accepts the same version twice, so a broken upload needs a new
version number (for example 0.1.1), not a re-upload.
