# All complete Go2 policies and training checkpoints

This archive preserves all **170 policy and checkpoint files** found under the training project's `logs/supermimic/` and `weights/` directories on 2026-11-07 (Asia/Shanghai), including historical training stages and backup checkpoints. The files total **6,769,095,205 bytes** (6.30 GiB). Each file was copied unchanged and checked against the source SHA-256 before committing and pushing.

Files retain their original paths beneath this directory. `logs/heightfields/` files are terrain caches, not policies, and are excluded. The three original `weights/` policies are included as well as the complete training checkpoints. Duplicate-content files retain their original filenames; Git stores identical content once.

## Download

Files are stored as ordinary Git blobs, so the archive is available through GitHub file links and normal Git cloning without Git LFS. GitHub rejects uploading new LFS objects to public forks; the archive was therefore pushed in batches below GitHub's push limit. Each individual file is below 100 MB.

For a smaller checkout of the source and the previously published videos, use a partial clone and sparse checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/zhshui/go2-supermimic.git
cd go2-supermimic
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/weights/complete/'
git checkout main
```

The default full clone includes all checkpoints. To download one checkpoint in the sparse checkout, add its original path under `weights/complete/` to your sparse-checkout patterns.

## Contents and provenance

[manifest.json](manifest.json) records each original path, size, source modification time and SHA-256. Full training checkpoints preserve the original actor, critic, world model, optimizer states and other metadata where present. Files that were actor-only policies or historical backups remain exactly as originally saved; they have not been relabeled as resumable training checkpoints.

The most recent full checkpoint is also available at [weights/locomotion/model_188500.pt](../locomotion/model_188500.pt), with a separate actor export and configuration. These are experimental results, not a claim that every checkpoint passed acceptance. See [the recorded rejected-trial status](../../experiments/local-training/GO2-LATEST-STATUS.md) and [the experiment archive](../../experiments/local-training/README.md).

## Checkpoint index

| Original file | Bytes |
| --- | ---: |
| [logs/supermimic/2026-11-02_19-27-50_stairs_llm/model_14000.pt](logs/supermimic/2026-11-02_19-27-50_stairs_llm/model_14000.pt) | 37,817,007 |
| [logs/supermimic/2026-11-02_19-27-50_stairs_llm/model_29000.pt](logs/supermimic/2026-11-02_19-27-50_stairs_llm/model_29000.pt) | 37,817,007 |
| [logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_12500_actor_lr_scale.pt](logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_12500_actor_lr_scale.pt) | 37,817,007 |
| [logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_4450_actor_lr_scale.pt](logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_4450_actor_lr_scale.pt) | 37,816,765 |
| [logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_8550_actor_lr_scale.pt](logs/supermimic/2026-11-02_19-27-50_stairs_llm/online-control/backups/before_8550_actor_lr_scale.pt) | 37,816,765 |
| [logs/supermimic/2026-11-02_21-39-26_mixed_commands_dr/model_29000.pt](logs/supermimic/2026-11-02_21-39-26_mixed_commands_dr/model_29000.pt) | 17,248,183 |
| [logs/supermimic/2026-11-02_22-23-46_mixed_commands_dr_60k/model_60000.pt](logs/supermimic/2026-11-02_22-23-46_mixed_commands_dr_60k/model_60000.pt) | 38,672,303 |
| [logs/supermimic/2026-11-03_10-45-41_mixed_gait_70k/model_60000.pt](logs/supermimic/2026-11-03_10-45-41_mixed_gait_70k/model_60000.pt) | 38,672,303 |
| [logs/supermimic/2026-11-03_10-45-41_mixed_gait_70k/model_70000.pt](logs/supermimic/2026-11-03_10-45-41_mixed_gait_70k/model_70000.pt) | 38,672,303 |
| [logs/supermimic/2026-11-03_11-50-59_gait_review_72000/model_70000.pt](logs/supermimic/2026-11-03_11-50-59_gait_review_72000/model_70000.pt) | 38,673,519 |
| [logs/supermimic/2026-11-03_11-56-19_gait_review_72000/model_70000.pt](logs/supermimic/2026-11-03_11-56-19_gait_review_72000/model_70000.pt) | 38,673,583 |
| [logs/supermimic/2026-11-03_12-23-24_gait_review_76000/model_76000.pt](logs/supermimic/2026-11-03_12-23-24_gait_review_76000/model_76000.pt) | 38,673,647 |
| [logs/supermimic/2026-11-03_12-47-16_gait_review_78000/model_76000.pt](logs/supermimic/2026-11-03_12-47-16_gait_review_78000/model_76000.pt) | 38,673,711 |
| [logs/supermimic/2026-11-03_13-07-13_gait_review_80000/model_80000.pt](logs/supermimic/2026-11-03_13-07-13_gait_review_80000/model_80000.pt) | 38,673,711 |
| [logs/supermimic/2026-11-03_13-22-37_gait_review_82000/model_80000.pt](logs/supermimic/2026-11-03_13-22-37_gait_review_82000/model_80000.pt) | 38,673,711 |
| [logs/supermimic/2026-11-03_13-48-00_gait_review_82000/model_80000.pt](logs/supermimic/2026-11-03_13-48-00_gait_review_82000/model_80000.pt) | 38,673,711 |
| [logs/supermimic/2026-11-03_17-14-56_gait_review_116000/model_116000.pt](logs/supermimic/2026-11-03_17-14-56_gait_review_116000/model_116000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_17-25-56_gait_review_118000/model_116000.pt](logs/supermimic/2026-11-03_17-25-56_gait_review_118000/model_116000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_17-36-56_gait_review_120000/model_120000.pt](logs/supermimic/2026-11-03_17-36-56_gait_review_120000/model_120000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_17-48-04_gait_review_122000/model_120000.pt](logs/supermimic/2026-11-03_17-48-04_gait_review_122000/model_120000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_17-49-08_gait_review_122000/model_120000.pt](logs/supermimic/2026-11-03_17-49-08_gait_review_122000/model_120000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_20-53-37_gait_review_124000/model_124000.pt](logs/supermimic/2026-11-03_20-53-37_gait_review_124000/model_124000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_21-10-23_gait_review_126000/model_124000.pt](logs/supermimic/2026-11-03_21-10-23_gait_review_126000/model_124000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_22-37-02_gait_review_134000/model_134000.pt](logs/supermimic/2026-11-03_22-37-02_gait_review_134000/model_134000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-03_23-08-07_gait_review_136000/model_134000.pt](logs/supermimic/2026-11-03_23-08-07_gait_review_136000/model_134000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_00-39-21_gait_review_138000/model_138000.pt](logs/supermimic/2026-11-04_00-39-21_gait_review_138000/model_138000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_00-59-42_gait_review_140000/model_138000.pt](logs/supermimic/2026-11-04_00-59-42_gait_review_140000/model_138000.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_00-59-42_gait_review_140000/model_139500.pt](logs/supermimic/2026-11-04_00-59-42_gait_review_140000/model_139500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_11-37-01_gait_review_141500/model_139500.pt](logs/supermimic/2026-11-04_11-37-01_gait_review_141500/model_139500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_11-37-01_gait_review_141500/model_141500.pt](logs/supermimic/2026-11-04_11-37-01_gait_review_141500/model_141500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_11-58-47_gait_review_143500/model_141500.pt](logs/supermimic/2026-11-04_11-58-47_gait_review_143500/model_141500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_11-58-47_gait_review_143500/model_143500.pt](logs/supermimic/2026-11-04_11-58-47_gait_review_143500/model_143500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_12-18-58_gait_review_145500/model_143500.pt](logs/supermimic/2026-11-04_12-18-58_gait_review_145500/model_143500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_12-18-58_gait_review_145500/model_145500.pt](logs/supermimic/2026-11-04_12-18-58_gait_review_145500/model_145500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_14-16-49_gait_review_147500/model_145500.pt](logs/supermimic/2026-11-04_14-16-49_gait_review_147500/model_145500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_14-16-49_gait_review_147500/model_147500.pt](logs/supermimic/2026-11-04_14-16-49_gait_review_147500/model_147500.pt) | 38,674,209 |
| [logs/supermimic/2026-11-04_14-34-41_gait_review_149500/model_147500.pt](logs/supermimic/2026-11-04_14-34-41_gait_review_149500/model_147500.pt) | 38,675,169 |
| [logs/supermimic/2026-11-04_14-34-41_gait_review_149500/model_149500.pt](logs/supermimic/2026-11-04_14-34-41_gait_review_149500/model_149500.pt) | 38,675,169 |
| [logs/supermimic/2026-11-04_14-57-14_gait_review_151500/model_149500.pt](logs/supermimic/2026-11-04_14-57-14_gait_review_151500/model_149500.pt) | 38,675,169 |
| [logs/supermimic/2026-11-04_14-57-14_gait_review_151500/model_151500.pt](logs/supermimic/2026-11-04_14-57-14_gait_review_151500/model_151500.pt) | 38,675,169 |
| [logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_151500.pt](logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_151500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_152000.pt](logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_152000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_152500.pt](logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_152500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_153000.pt](logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_153000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_153500.pt](logs/supermimic/2026-11-04_15-31-42_gait_review_153500/model_153500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_153500.pt](logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_153500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_154000.pt](logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_154000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_154500.pt](logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_154500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_155000.pt](logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_155000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_155500.pt](logs/supermimic/2026-11-04_15-57-59_gait_review_155500/model_155500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_155500.pt](logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_155500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_156000.pt](logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_156000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_156500.pt](logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_156500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_157000.pt](logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_157000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_157500.pt](logs/supermimic/2026-11-04_19-34-43_gait_review_157500/model_157500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_157500.pt](logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_157500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_158000.pt](logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_158000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_158500.pt](logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_158500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_159000.pt](logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_159000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_159500.pt](logs/supermimic/2026-11-04_19-57-34_gait_review_159500/model_159500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_159500.pt](logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_159500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_160000.pt](logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_160000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_160500.pt](logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_160500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_161000.pt](logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_161000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_161500.pt](logs/supermimic/2026-11-04_20-17-44_gait_review_161500/model_161500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_161500.pt](logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_161500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_162000.pt](logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_162000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_162500.pt](logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_162500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_163000.pt](logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_163000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_163500.pt](logs/supermimic/2026-11-04_20-37-37_gait_review_163500/model_163500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_163500.pt](logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_163500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_164000.pt](logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_164000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_164500.pt](logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_164500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_165000.pt](logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_165000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_165500.pt](logs/supermimic/2026-11-04_20-57-27_gait_review_165500/model_165500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_165500.pt](logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_165500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_166000.pt](logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_166000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_166500.pt](logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_166500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_167000.pt](logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_167000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_167500.pt](logs/supermimic/2026-11-04_21-17-22_gait_review_167500/model_167500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_167500.pt](logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_167500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_168000.pt](logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_168000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_168500.pt](logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_168500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_169000.pt](logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_169000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_169500.pt](logs/supermimic/2026-11-04_21-39-28_gait_review_169500/model_169500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_169500.pt](logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_169500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_170000.pt](logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_170000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_170500.pt](logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_170500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_171000.pt](logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_171000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_171500.pt](logs/supermimic/2026-11-04_22-07-16_gait_review_171500/model_171500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_171500.pt](logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_171500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_172000.pt](logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_172000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_172500.pt](logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_172500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_173000.pt](logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_173000.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_173500.pt](logs/supermimic/2026-11-05_18-21-23_gait_review_173500/model_173500.pt) | 38,675,233 |
| [logs/supermimic/2026-11-05_18-43-10_gait_review_175500/model_173500.pt](logs/supermimic/2026-11-05_18-43-10_gait_review_175500/model_173500.pt) | 14,711,753 |
| [logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_173500.pt](logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_173500.pt) | 14,711,753 |
| [logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_174000.pt](logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_174000.pt) | 44,087,649 |
| [logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_174500.pt](logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_174500.pt) | 44,087,649 |
| [logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_175000.pt](logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_175000.pt) | 44,087,649 |
| [logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_175500.pt](logs/supermimic/2026-11-05_18-47-18_gait_review_175500/model_175500.pt) | 44,087,649 |
| [logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_175500.pt](logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_175500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_176000.pt](logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_176000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_176500.pt](logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_176500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_177000.pt](logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_177000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_177500.pt](logs/supermimic/2026-11-05_19-15-45_gait_review_177500/model_177500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_177500.pt](logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_177500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_178000.pt](logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_178000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_178500.pt](logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_178500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_179000.pt](logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_179000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_179500.pt](logs/supermimic/2026-11-05_19-36-38_gait_review_179500/model_179500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_179500.pt](logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_179500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_180000.pt](logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_180000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_180500.pt](logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_180500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_181000.pt](logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_181000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_181500.pt](logs/supermimic/2026-11-05_20-43-53_gait_review_181500/model_181500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_181500.pt](logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_181500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_182000.pt](logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_182000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_182500.pt](logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_182500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_183000.pt](logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_183000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_183500.pt](logs/supermimic/2026-11-05_21-19-21_gait_review_183500/model_183500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_181500.pt](logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_181500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_182000.pt](logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_182000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_182500.pt](logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_182500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_183000.pt](logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_183000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_183500.pt](logs/supermimic/2026-11-05_21-48-25_gait_review_183500/model_183500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_181500.pt](logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_181500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_182000.pt](logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_182000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_182500.pt](logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_182500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_183000.pt](logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_183000.pt) | 44,087,969 |
| [logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_183500.pt](logs/supermimic/2026-11-05_22-13-32_gait_review_183500/model_183500.pt) | 44,087,969 |
| [logs/supermimic/2026-11-06_14-16-38_gait_review_swing_183500/model_181500.pt](logs/supermimic/2026-11-06_14-16-38_gait_review_swing_183500/model_181500.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_181500.pt](logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_181500.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_182000.pt](logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_182000.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_182500.pt](logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_182500.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_183000.pt](logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_183000.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_183500.pt](logs/supermimic/2026-11-06_14-30-20_gait_review_swing_183500/model_183500.pt) | 44,088,033 |
| [logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_181500.pt](logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_181500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_182000.pt](logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_182000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_182500.pt](logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_182500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_183000.pt](logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_183000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_183500.pt](logs/supermimic/2026-11-06_15-19-27_gait_review_swing_calf_183500/model_183500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_181500.pt](logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_181500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_182000.pt](logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_182000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_182500.pt](logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_182500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_183000.pt](logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_183000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_183500.pt](logs/supermimic/2026-11-06_16-08-39_gait_review_calf_only_183500/model_183500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_183500.pt](logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_183500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_184000.pt](logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_184000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_184500.pt](logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_184500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_185000.pt](logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_185000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_185500.pt](logs/supermimic/2026-11-06_17-41-10_gait_review_calf_acc1p5_185500/model_185500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_183500.pt](logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_183500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_184000.pt](logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_184000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_184500.pt](logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_184500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_185000.pt](logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_185000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_185500.pt](logs/supermimic/2026-11-06_19-11-46_gait_review_calf_acc1p5_lrhalf_185500/model_185500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_185500.pt](logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_185500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_186000.pt](logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_186000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_186500.pt](logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_186500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_187000.pt](logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_187000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_187500.pt](logs/supermimic/2026-11-06_20-18-29_gait_review_calf_acc1p5_lrhalf_187500/model_187500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_186500.pt](logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_186500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_187000.pt](logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_187000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_187500.pt](logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_187500.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_188000.pt](logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_188000.pt) | 44,088,097 |
| [logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_188500.pt](logs/supermimic/2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500/model_188500.pt) | 44,088,097 |
| [weights/flat.pt](weights/flat.pt) | 1,388,559 |
| [weights/obstacles.pt](weights/obstacles.pt) | 1,553,572 |
| [weights/terrain.pt](weights/terrain.pt) | 2,180,887 |
