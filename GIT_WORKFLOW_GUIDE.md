# Git Workflow Guide for Multi-Agent Development

## Current Situation
- **Branch**: `fix-polar-plot-positioning`
- **Modified file**: `app.py` (lines ~150-165)
- **Status**: Uncommitted changes

## Best Practice Workflow

### Step 1: Commit Your Current Work
```powershell
cd "C:\Dev\Python\Estacion metereologica\Fire risk dashboard"
git add app.py
git commit -m "Fix polar plot positioning - move plot inside colA column context"
```

### Step 2: Before Starting New Work with Another Agent

**Option A: Push your branch (recommended)**
```powershell
git push -u origin fix-polar-plot-positioning
```
This backs up your work and makes it visible to others.

**Option B: Just commit locally**
If you're working solo, local commits are fine. Push when ready to merge.

### Step 3: Starting New Work

**For other agents working on different files:**
- They can create new branches from `main` or your branch
- No conflicts expected if files don't overlap

**For other agents working on the same file (`app.py`):**
- They should create a branch from `main` (not your branch)
- Or wait until you merge your branch first
- If they modify different sections, Git usually auto-merges

## Conflict Scenarios

### Scenario 1: No Conflict (Different Files)
```
Your branch: modifies app.py
Other branch: modifies visualizations.py
Result: ✅ Auto-merge, no conflicts
```

### Scenario 2: No Conflict (Same File, Different Sections)
```
Your branch: modifies app.py lines 150-165
Other branch: modifies app.py lines 200-250
Result: ✅ Auto-merge, no conflicts (Git is smart!)
```

### Scenario 3: Conflict (Same File, Overlapping Lines)
```
Your branch: modifies app.py lines 150-165
Other branch: modifies app.py lines 160-170
Result: ⚠️ CONFLICT - needs manual resolution
```

## What Happens During Merge with Conflicts

When you merge branches with conflicts:

1. **Git detects the conflict:**
   ```
   Auto-merging app.py
   CONFLICT (content): Merge conflict in app.py
   ```

2. **Git marks the conflict in the file:**
   ```python
   <<<<<<< HEAD (your current branch)
   # Your code here
   =======
   # Other branch's code here
   >>>>>>> other-branch-name
   ```

3. **You resolve manually:**
   - Edit the file to keep the code you want
   - Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
   - Test that everything works

4. **Complete the merge:**
   ```powershell
   git add app.py
   git commit -m "Merge branch 'other-branch' - resolved conflicts"
   ```

## Recommended Strategy for Your Project

### Strategy 1: Sequential Work (Safest)
1. Commit and merge your branch first
2. Then start new work from updated `main`
3. **Pros**: No conflicts, clear history
4. **Cons**: Slower, can't work in parallel

### Strategy 2: Parallel Work with Communication
1. Commit your branch (don't merge yet)
2. Tell other agents which files you modified
3. They avoid those files or coordinate changes
4. **Pros**: Faster, parallel development
5. **Cons**: Requires coordination

### Strategy 3: Feature Branches (Best for Teams)
1. Each feature gets its own branch
2. Work in parallel on different features
3. Merge to `main` one at a time
4. **Pros**: Organized, scalable
5. **Cons**: Need to manage multiple branches

## Checking for Potential Conflicts

Before merging, you can check what would conflict:

```powershell
# See what files changed in your branch
git diff main..fix-polar-plot-positioning --name-only

# See what would conflict (dry-run merge)
git merge --no-commit --no-ff other-branch-name
# If conflicts, abort: git merge --abort
```

## Your Current Situation

**What to do now:**
1. ✅ Commit your changes (recommended before other work)
2. ✅ Push your branch (optional but good practice)
3. ✅ When ready, merge to `main`:
   ```powershell
   git checkout main
   git merge fix-polar-plot-positioning
   git push origin main
   ```

**For other agents:**
- If they work on `visualizations.py`, `risk_calculator.py`, etc. → No conflicts
- If they work on `app.py` but different sections → Usually no conflicts
- If they work on same `app.py` lines → Will have conflicts (need coordination)

## Conflict Resolution Tips

1. **Use a merge tool**: `git mergetool` opens a visual diff tool
2. **Keep both changes**: Sometimes you need code from both branches
3. **Test after resolving**: Always test after resolving conflicts
4. **Ask for help**: If unsure, don't guess - ask or review both versions

## Example: Safe Parallel Work

```
Timeline:
Day 1: You commit "fix-polar-plot-positioning" (modifies app.py lines 150-165)
Day 2: Other agent creates "add-export-feature" branch
       - Modifies app.py lines 280-300 (different area)
       - No conflict! ✅
Day 3: You merge your branch to main
Day 4: Other agent merges their branch to main
       - Git auto-merges both changes ✅
```

## Summary

- ✅ **Commit early and often** - creates checkpoints
- ✅ **Different files = safe** - no conflicts
- ✅ **Same file, different areas = usually safe** - Git auto-merges
- ⚠️ **Same file, same area = conflicts** - need manual resolution
- ✅ **Conflicts are fixable** - Git helps you resolve them
- ✅ **Test after merging** - always verify everything works

