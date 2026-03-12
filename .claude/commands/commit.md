Review the staged changes and write a professional git commit message.

First, run:
```
git diff --staged
```

Then produce a commit message following Conventional Commits format:

**Subject line** (max 72 chars):
`type(scope): concise description in imperative mood`

Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`

**Body** (if non-trivial): explain *what* changed and *why*, not how.

**Footer** (if applicable): note breaking changes or issue references.

Finally, run `git commit -m` with the message you wrote.
```

## Usage

In any Claude Code session inside Cursor's terminal:
```
/commit
