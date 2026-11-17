# Security

This is a portfolio project. It runs locally and is not deployed, so there is no live service to attack.

## Reporting a problem

If you find a security issue (for example a leaked secret, a guardrail bypass, or an unsafe dependency), please use GitHub's private vulnerability reporting on this repository rather than opening a public issue.

## How secrets are handled

- The only secret is `ANTHROPIC_API_KEY`, read from a local `.env` file that is git-ignored.
- The unit tests use a fake client and need no key. Tests that call the real API are opt-in (`-m integration`).
- The CI workflow has read-only permissions, pins every action to a full commit SHA, and does not keep the checkout token.
- Dependabot checks the pinned Python packages and the workflow actions every week.
