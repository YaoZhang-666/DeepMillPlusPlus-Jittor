# Pretrained checkpoint

The model checkpoint is intentionally excluded from this source repository.
Download the published DeepMill++ checkpoint and place it in this directory:

```bash
curl -L --fail \
  -o DeepM/pretrained/deepm_00100.model.pth \
  https://raw.githubusercontent.com/YaoZhang-666/DeepMillPlusPlus/main/DeepM/pretrained/deepm_00100.model.pth
```

The final location must be:

```text
DeepM/pretrained/deepm_00100.model.pth
```

The inference commands in the repository require that exact file path.
