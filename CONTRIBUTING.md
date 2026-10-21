# Contributing

Thank you for helping. Changes come in as pull requests against `main`.

## Making a change

1. Fork the repository and create a branch for your change.
2. Keep one change per pull request, and say in the description what it fixes and how you tested it.
3. Run the repository's tests before you push, as its README describes.

4. Open a pull request. A maintainer reviews it, and the checks must pass before it is merged.

## Sign off every commit

Every commit in a pull request must carry a sign-off. It states that you wrote the change, or
otherwise have the right to submit it under this project's licence (Apache 2.0). The full text
is the Developer Certificate of Origin: <https://developercertificate.org>.

Sign off by committing with `-s`, which adds a line with your name and email:

```
git commit -s -m "Fix the folder picker"
```

```
Signed-off-by: Your Name <you@example.com>
```

Use your real name, and the email in your Git settings. A check on the pull request fails if any
commit has no sign-off.

**To add a missing sign-off:**

- Only the last commit: `git commit --amend -s --no-edit`, then `git push --force-with-lease`.
- Several commits: `git rebase --signoff HEAD~3` (3 being how many commits your branch adds),
  then `git push --force-with-lease`.

Force-push only your own pull request branch.

## Reporting a vulnerability

Please do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).

## Licence

By contributing, you agree that your contribution is licensed under the Apache License 2.0, the same as
the rest of the project. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
