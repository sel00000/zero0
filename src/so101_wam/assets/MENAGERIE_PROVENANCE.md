# Vendored SO-101 MuJoCo model

`mujoco_menagerie/robotstudio_so101` is an unmodified copy of Google
DeepMind's MuJoCo Menagerie model at commit
`da76818e269b82289eba39808e2fb91d679d6994`.

- Upstream: <https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101>
- Retrieved: 2026-08-31
- License: Apache-2.0; see the vendored `LICENSE` file
- Upstream minimum MuJoCo version: 3.1.3

The local `dual_so101_wam.xml` scene attaches two prefixed instances of that
model to a fixed humanoid torso. The scene itself belongs to this prototype;
the vendored meshes and upstream SO-101 MJCF retain their upstream license.
