# 2D RPE Operators

## Build iRPE operators implemented by CUDA.
Although iRPE can be implemented by PyTorch native functions, the backward speed of PyTorch index function is very slow. We implement CUDA operators for more efficient training and recommend to build it. `nvcc` is necessary to build CUDA operators.
```bash
cd rpe_ops/
python setup.py install --user
```

## rpe\_index
The function [`rpe_index`](./rpe_index.py#L5) is equal to
```python
def rpe_index(input, index):
    '''Y[b, h, i, j] = input[b, h, i, index[i, j]]

    Parameters
    ----------
    input: torch.Tensor, float32
        The shape is (B, H, L_query, num_buckets)
    index: torch.Tensor, int32
        The shape is (L_query, L_key)

    where B is the batch size, and H is the number of attention heads.

    Returns
    -------
    Y: torch.Tensor, float32
        The shape is (B, H, L_query, L_key)
    '''
    L_query, L_key = index.shape
    num_buckets = input.size(-1)
    B = len(input)
    offset = torch.arange(0, L_query * num_buckets, num_buckets).view(-1, 1)
    return input.flatten(2)[:, :, (index + offset).flatten()].view(B, -1, L_query, L_key)
```
