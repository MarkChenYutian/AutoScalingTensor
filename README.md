<p align="center">
    <img width="200" alt="Screenshot 2024-05-05 at 2 17 14 AM" src="https://github.com/MarkChenYutian/AutoScalingTensor/assets/47029019/48cc04d4-eb20-4a40-8589-251cd75bb2c7">
</p>

<h1 align="center">AutoScalingTensor</h1>


Automatic-resized PyTorch Tensor that supports all Pytorch API with almost zero abstraction cost. Amortized O(1) `torch.cat` along specific dimension. 

## Interface

### Create an AutoScalingTensor

```python
AutoScalingTensor(
    shape: tuple[int, ...],
    grow_on: int,
    init_tensor: torch.Tensor | None = None,
    init_val: int | float | None = None
)
```

* `shape`   - the shape of tensor, all dimensions except the `grow_on`-dim will remain the same shape once created.
* `grow_on` - the dimension to append data on.
* `init_tensor` - (Optional) Use some existing tensor as the initial storage, `init_tensor`'s shape must match `shape` argument.
* `init_val` - (Optional) The initial value for tensor. If not provided, tensor values will be uninitilized to maximize performance (using `torch.empty`). Otherwise, underlying tensor will always be initilized with `init_val` filled (using `torch.fill`).

### Add data (efficiently) to AutoScalingTensor

```python
AutoScalingTensor.push(x: torch.Tensor) -> None
```

* `x` - the shape should be the same as `shape` provided to `__init__` except on `grow_on`-dim.

### Retrieve data efficiently from AutoScalingTensor

```python
acc: AutoScalingTensor  # Some AutoScalingTensor

acc_tensor_view: torch.Tensor = acc.tensor      # Get a view of all data with *.tensor
acc_partial_view: torch.Tensor = acc[..., :100] # All indexing methods are supported, advanced or naive.
```

## Efficient Tensor Accumulation

Concatenating `1 x 3` tensor into an `accumulator` tensor iteratively for `400,000` iterations, `AutoScalingTensor` only need `<2sec` while naive `torch.Tensor` need `35sec+` on Apple M2 chip laptop.

| Num_iter | Time (`AutoScalingTensor`) | Time (`torch.Tensor`) | 
|--|--|--|
| 200,000 | 0.9995s | 8.621s | 
| 300,000 | 1.4492s | 21.6432s |
| 400,000 | 1.8861s | 37.9394s |

Benchmarking Code:

```python
@timing
def test_autoscale(niter: int) -> AutoScalingTensor:
    accumulator = AutoScalingTensor(shape=(8, 3), grow_on=0)
    data = torch.tensor([[1, 2, 3]])
    for idx in range(niter):
        accumulator.push(data * idx)
    return accumulator

@timing
def test_naive_cat(niter: int) -> torch.Tensor:
    accumulator = torch.zeros((0, 3))
    data = torch.tensor([[1, 2, 3]])
    for idx in range(niter):
        accumulator = torch.cat([accumulator, data * idx], dim=0)
    return accumulator

A = test_autoscale(200000)
B = test_naive_cat(200000)
```

## Fully Compatible with `torch.Tensor` 

`AutoScalingTensor` can merge into existing PyTorch projects with **ZERO COST**. `AutoScalingTensor` supports all PyTorch APIs, chain-calling, advanced indexing, etc. just like a torch.Tensor do.

```python
>>> from AutoScaleTensor import AutoScalingTensor
>>> import torch
>>> A = AutoScalingTensor((1, 3), grow_on=0)
>>> A.push(torch.tensor([[0., 0., 0.], [10., 20., 30.]]))
>>> A
AutoScalingTensor(alloc=4, actual=2, 
        data=tensor([[ 0.,  0.,  0.],
        [10., 20., 30.]])
)
>>> A[0]
tensor([0., 0., 0.])
>>> A[0] = 3.
>>> A
AutoScalingTensor(alloc=4, actual=2, 
        data=tensor([[ 3.,  3.,  3.],
        [10., 20., 30.]])
)
>>> A[1].mean()
tensor(20.)
```
