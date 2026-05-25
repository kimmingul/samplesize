# Release process

This project ships to PyPI via the **PyPI Trusted Publisher** flow.
There are no long-lived API tokens stored as secrets — PyPI mints a
short-lived OIDC token for the GitHub Actions job at publish time.

## One-time setup (already done once, reference only)

1. Create a pending publisher on PyPI at
   <https://pypi.org/manage/account/publishing/>:
   - PyPI Project Name: `samplesize`
   - Owner: `kimmingul`
   - Repository name: `samplesize-copilot`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. In GitHub repo settings → Environments, create an environment named
   `pypi`. (Reviewer/approval is optional; the trust is asserted by the
   OIDC issuer + repo + workflow + environment tuple.)

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml
sed -i '' 's/version = "0.1.0"/version = "0.2.0"/' pyproject.toml

# 2. Update the CHANGELOG (if present)
$EDITOR CHANGELOG.md

# 3. Commit + tag + push
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0"
git tag v0.2.0
git push origin main --tags
```

The `release.yml` workflow fires on the `v*` tag, builds the sdist +
wheel, and publishes them via the trusted-publisher action. The new
version appears on <https://pypi.org/project/samplesize/> within
a couple of minutes.

## Verifying a release

```bash
pip install samplesize==<new-version>
python -c "import samplesize; print(samplesize.__version__)"
samplesize doctor
```

## Manual emergency upload

If the trusted-publisher path fails (e.g., OIDC misconfiguration), a
maintainer with an API token can upload from a clean checkout:

```bash
python -m build
twine upload dist/*
```

This requires `twine` and a `~/.pypirc` with a project-scoped API token.
Prefer the trusted-publisher path for routine releases.
