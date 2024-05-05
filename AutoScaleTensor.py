import torch
import math
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    # Since extending torch.Tensor class using __torch_function__ is not supported by 
    # static type checker like MyPy and Pyright, we use this dummy class to fool the 
    # static analysis tool that AutoScalingTensor behaves like a torch.Tensor.
    # https://github.com/pytorch/pytorch/issues/75568
    # https://github.com/pytorch/pytorch/pull/75484
    # 
    # Due to the auto attribute delegation to torch.Tensor in the AutoScalingTensor.__getattribute__(...)
    # this version visible to type hinting actually matches all valid usages of the AutoScalingTensor
    # so there is no significant discrepency between static analysis bahavior and actual runtime result.
    class AutoScalingTensor(torch.Tensor):
        def __init__(self, 
                     shape: torch.Size | Sequence[int] | None, 
                     grow_on: int, 
                     init_tensor: torch.Tensor | None = None) -> None: ...
        def __new__(cls, *args, **kwargs) -> "AutoScalingTensor": ...
        def push(self, x: torch.Tensor) -> None: ...
else:
    class AutoScalingTensor:
        def __init__(self, 
                    shape: torch.Size | Sequence[int] | None, 
                    grow_on: int, 
                    init_tensor: torch.Tensor | None = None
                    ) -> None:
            self.grow_on = grow_on
            self.current_size = 0
            if shape is not None:
                self._tensor = torch.empty(shape)
            else:
                assert init_tensor is not None
                self._tensor = init_tensor
        
        def _scale_up_to(self, size: int):
            grow_to = 2 ** math.ceil(math.log2(size + 1))
            orig_shape = list(self._tensor.shape)
            orig_shape[self.grow_on] = grow_to
            new_storage = torch.empty(orig_shape)
            new_storage.narrow(dim=self.grow_on, start=0, length=self.current_size).copy_(
                self._tensor.narrow(dim=self.grow_on, start=0, length=self.current_size)
            )
            self._tensor = new_storage
        
        @property
        def _curr_max_size(self) -> int:
            return self._tensor.size(self.grow_on)
        
        @property
        def tensor(self) -> torch.Tensor:
            return self._tensor.narrow(dim=self.grow_on, start=0, length=self.current_size)
        
        def __repr__(self) -> str:
            return f"AutoScalingTensor(alloc={self._curr_max_size}, actual={self.current_size}, \n\tdata={self.tensor}\n)"

        def push(self, x: torch.Tensor) -> None:
            data_size = x.size(self.grow_on)
            
            if self.current_size + data_size >= self._curr_max_size:
                self._scale_up_to(self.current_size + data_size)
            assert self.current_size < self._curr_max_size
            
            self._tensor.narrow(dim=0, start=self.current_size, length=data_size).copy_(x)
            self.current_size += data_size
        
        def __getitem__(self, slice):
            return self.tensor.__getitem__(slice)
        
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            args = [data.tensor if isinstance(data, AutoScalingTensor) else data for data in args]
            return func(*args, **kwargs)

        # A further enhancement - we want AutoScale to behave exactly like the tensor it contains
        def __getattribute__(self, name: str):
            if name in ['tensor', '__class__']:
                return object.__getattribute__(self, name)
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                # If it's not a class attribute, delegate to the tensor
                tensor = object.__getattribute__(self, 'tensor')
                return getattr(tensor, name)
