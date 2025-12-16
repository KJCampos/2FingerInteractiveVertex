# 2FingerInteractiveVertex quick notes

## HUD CAD scope
The current `hud_cad.py` intentionally stays lean and focused on the input/render loop rather than sprawling helpers. The file keeps only the glue needed to:
- track fingertip pointers and pinch gestures
- accumulate snapped line segments into a loop
- triangulate/extrude a finished polygon
- expose mouse rotation/reset hooks and mesh handoff

Utility math/mesh logic lives in `mesh_25d.py`, while physics/render concerns are kept in `physics3d_ti.py` and `renderer3d_ti.py`. This separation aims to avoid over-coding `hud_cad.py` while keeping the CAD flow readable.

## Testing and sharing each version
- Check worktree state with `git status` and inspect diffs with `git diff` before committing.
- Run a quick syntax pass with `python -m py_compile app.py hud_cad.py voice_cmd.py mesh_25d.py physics3d_ti.py renderer3d_ti.py`.
- Commit with a meaningful message: `git commit -am "Your summary"`.
- If collaborating remotely, push your branch (e.g., `git push origin <branch>`), then open a pull request.
- To trial a specific commit locally, use `git checkout <commit-sha>` and run `python app.py` to exercise the full pipeline.
