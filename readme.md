# TensorRT<sup>\*</sup> in OpenFOAM

1. Compile the trtInfer with

```c
wmake libso
```

2. Compile the inferFoam with

```c
wmake
```

3. The testCase shows that the inference works(taking in_1 and in_2 and produce 4 out scalar fields).

<sup>\*This demo works with tensorrt 5.
