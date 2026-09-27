# Data Directory

The repository does not include full benchmark tables or feature matrices by
default. Put local or released data under `data--final/` using the layout shown
in the root README.

Minimum split columns:

```text
fasta_id,sequence,label_id
```

Recommended additional columns:

```text
length,label,task_label,source_dbs,source_files
```

Optional MD reproduction inputs can be placed locally under:

```text
data/md/p1/
  production_100ns.tpr
  production_100ns_centered.xtc
```

Trajectory/topology files are intentionally not tracked by this open-release
package. Keep large MD outputs in external storage or release assets.
