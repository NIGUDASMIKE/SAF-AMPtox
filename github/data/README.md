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
