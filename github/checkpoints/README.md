# Checkpoints

Place trained SAF-AMPTox fusion checkpoints under `checkpoints/fusion/`.

Expected names for the main ensemble:

```text
amp__cross_attention_residual__seed13.pt
amp__cross_attention_residual__seed29.pt
amp__cross_attention_residual__seed47.pt
tox__cross_attention_residual__seed13.pt
tox__cross_attention_residual__seed29.pt
tox__cross_attention_residual__seed47.pt
```

Large EvoDiff weights are intentionally not tracked in this code release.
