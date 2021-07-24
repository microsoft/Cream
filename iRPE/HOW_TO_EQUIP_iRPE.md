# How to equip iRPE ?

The implementation of iRPE (image relative position encoding) contains two parts, namely python part `irpe.py` and C++/CUDA part `rpe_ops`. The python code `irpe.py` is the basic part to implement the four kinds of relative position encoding mappings, and the C++/CUDA code `rpe_ops` accelerate the forward and backward procedure. We should copy two parts to the project directory which need iRPE.

Current implementation supports variable input resolution and non-square input.

- Step. 1 - Copy the iRPE files

Copy the file `irpe.py` and the directory `rpe_ops` to the project directory.

- Step. 2 - Create the configuration of RPE

    - [Example in DeiT: rpe\_models.py#L14-L21](./DeiT-with-iRPE/rpe_models.py#L14-L21)
    - [Example in DETR: models/transformer.py#L63-L69](./DETR-with-iRPE/models/transformer.py#L63-L69)

```python
from irpe import get_rpe_config
rpe_config = get_rpe_config(
    ratio=1.9,
    method="product",
    mode='ctx',
    shared_head=True,
    skip=1,
    rpe_on='k',
)
```

The meaning of arguments could be seen in [`help(get_rpe_config)`](./DeiT-with-iRPE/irpe.py#L823-L855).

- Step. 3 - Build the instance of RPE modules

    - [Example in DeiT: rpe\_vision\_transformer.py#L63-L66](./DeiT-with-iRPE/rpe_vision_transformer.py#L63-L66)
    - [Example in DETR: models/rpe\_attention/multi\_head\_attention.py#L94-L97](./DETR-with-iRPE/models/rpe_attention/multi_head_attention.py#L94-L97)

```python
from irpe import build_rpe

def __init__(self, ...):
    ...
    # image relative position encoding
    self.rpe_q, self.rpe_k, self.rpe_v = \
        build_rpe(rpe_config,
                  head_dim=head_dim,
                  num_heads=num_heads)
```
`build_rpe` should be called in the function `__init__` of a `nn.Module`.

- Step. 4 - Add RPE on keys, queries and values 

    - [Example in DeiT: rpe\_vision\_transformer.py#L77-L92](./DeiT-with-iRPE/rpe_vision_transformer.py#L77-L92)
    - [Example in DETR: rpe\_vision\_transformer.py#L327-L376](./DETR-with-iRPE/models/rpe_attention/rpe_attention_function.py#L327-L376)

In the `forward` function, we consider relative position encodings as a bias on `attn` and `attn @ v`.
```python
def forward(self, ...):
    ...
    attn = (q @ k.transpose(-2, -1))

    # image relative position on keys
    if self.rpe_k is not None:
        attn += self.rpe_k(q)

    # image relative position on queries
    if self.rpe_q is not None:
        attn += self.rpe_q(k * self.scale).transpose(2, 3)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    out = attn @ v

    # image relative position on values
    if self.rpe_v is not None:
        out += self.rpe_v(attn)

    x = out.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

Notice that the shapes of `q`, `k` and `v` are all `(B, H, L, head_dim)`, where `B` is batch size, `H` is the number of heads, `L` is the sequence length, equal to `height * width` (+1 if class token exists). `head_dim` is the dimension of each head.

- Step. 5 [Optional, Recommend] - Build C++/CUDA operators for iRPE

Although iRPE can be implemented by PyTorch native functions, the backward speed of PyTorch index function is very slow. We implement CUDA operators for more efficient training and recommend to build it.
`nvcc` is necessary to build CUDA operators.
```bash
cd rpe_ops/
python setup.py install --user
```
