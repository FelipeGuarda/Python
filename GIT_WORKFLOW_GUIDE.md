# Git Workflow Guide for Home–Work Solo Dev

This guide explains **where your work lives** (local vs remote, main vs feature branches) and a **fast, repeatable workflow** for moving work between computers.

---

## Mental Model: Where Is My Work?

For any branch name (for example `main` or `fix-polar-plot-positioning`), think of three layers:

- `origin/main`: The branch on GitHub. This is the **source of truth** shared between home and work.
- Local `main`: The same branch on each computer. It can be **ahead of** or **behind** `origin/main`.
- Local feature branches: `fix-polar-plot-positioning`, `new-legend`, etc. Created from a local branch and optionally pushed to `origin/<branch>`.

A typical life cycle for a feature:

1. Start from updated `main` on one machine.
2. Create a feature branch (e.g. `fix-polar-plot-positioning`) from `main`.
3. Commit on that branch locally; push to `origin/fix-polar-plot-positioning` when you want to move it between machines or back it up.
4. On another machine, fetch and check out that feature branch and continue working.
5. When done, merge the feature branch into `main`, push `main`, then delete the finished branch.

You can always inspect where things are with:

```bash
git branch -vv # shows which local branches track which remotes, ahead/behind
git log --oneline --graph --decorate --all | head
```

---

## Daily Start: Sync With GitHub

Always begin a session by making sure `main` matches GitHub:

```bash
git checkout main
git pull origin main
```

This ensures home, work, and GitHub all converge on the same `main` tip before you branch or continue work.

If you are about to resume an existing feature branch:

```bash
git checkout my-feature
git pull --ff-only # optional: fast-forward local branch to origin/my-feature
```

---

## Starting a New Feature

From an updated `main`:

```bash
git checkout main
git pull origin main # ensure main is current
git checkout -b my-feature # create feature branch from main
```

Now all commits for this feature stay on `my-feature` until you decide to merge.

---

## Working and Committing

While coding on a feature branch:

```bash
git status
git add <files>
git commit -m "Describe what changed"
```

Push to GitHub when you want the work available on the other machine or safely backed up:

**First push** (creates remote tracking branch):

```bash
git push -u origin my-feature
```

**Later pushes** on same branch:

```bash
git push
```

This creates `origin/my-feature` on GitHub, which you can then pull from your other computer.

---

## Switching Between Home and Work

When you move to another machine and want to continue the same feature:

1. **Sync main with GitHub:**

```bash
git checkout main
git pull origin main
```

2. **Make sure you see the remote feature branch:**

```bash
git fetch origin
```

3. **Check out the feature branch:**

```bash
git checkout my-feature # first time: creates local branch tracking origin/my-feature
```

4. **(Optional) fast-forward if needed:**

```bash
git pull --ff-only
```

Now you are on the same feature branch with the same commits, ready to continue work.

---

## Finishing a Feature: Merge Into main

Once the feature is ready and tested on one machine:

1. **Make sure main is up to date:**

```bash
git checkout main
git pull origin main
```

2. **Merge your feature branch into main:**

```bash
git merge --no-ff my-feature # or just git merge my-feature if you are fine with fast-forward
```

3. **Run tests, then publish to GitHub:**

```bash
git push origin main
```

At this point, all commits from `my-feature` live in `main` (both locally and on GitHub).

---

## Cleaning Up Finished Branches

After a successful merge to `main`, the feature branch becomes redundant. Best practice is to delete it to keep your branch list clean:

**Delete local branch** (only if it is fully merged):

```bash
git branch -d my-feature
```

**Delete remote branch on GitHub:**

```bash
git push origin --delete my-feature
```

Deleting the branch **does not delete the commits**. They are preserved in `main`’s history, and you can always find them via `git log` or by commit hash.

---

## Handling Build Artifacts (e.g. __pycache__)

Python generates `__pycache__` and `.pyc` files that should not be tracked by Git.

**One-time cleanup:**

**Add to .gitignore:**

```bash
echo "pycache/" >> .gitignore
echo "*.pyc" >> .gitignore
git add .gitignore
```

**Stop tracking existing cached files:**

```bash
git rm -r --cached **/pycache || true
git rm --cached $(git ls-files "*.pyc") || true
git commit -m "Stop tracking pycache and .pyc files"
git push origin main
```

After this, `__pycache__` and `.pyc` files will no longer appear as modified; they remain only as local artifacts.

---

## Conflict Scenarios (Short Version)

- Different files → no conflict; Git auto-merges.
- Same file, different areas → usually no conflict; Git can merge both changes.
- Same file, same lines → conflict; you must resolve manually.

During a conflict:

1. Git marks conflicts with `<<<<<<<`, `=======`, `>>>>>>>` in the file.
2. Edit the file to the desired final version and remove the markers.
3. Test, then:

```bash
git add <file>
git commit -m "Resolve merge conflict in <file>"
```

---

## Summary Checklist

- **Start session:** `git checkout main && git pull origin main`
- **New work:** `git checkout -b my-feature main`
- **Save progress:** `git add ... && git commit ... && git push`
- **Switch machine:** `git fetch origin && git checkout my-feature`
- **Finish feature:** merge into `main`, push `main`
- **Clean up:** delete local and remote feature branch after merge
