import numpy as np
import math

from typing import Sequence
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray


class AutoScalingArray(NDArrayOperatorsMixin):
    def __init__(self,
                 shape: Sequence[int] | None,
                 grow_on: int,
                 init_array: NDArray | None=None,
                 **kwargs
                ) -> None:
        super().__init__()
        self.grow_on = grow_on
        self.current_size = 0
        if shape is not None:
            self._array = np.empty(shape, **kwargs)
            self._curr_max_size = shape[grow_on]
        else:
            assert init_array is not None
            self._array = init_array
            self._curr_max_size = self._array.shape[grow_on]
    
    @staticmethod
    def _numpy_narrow_equiv(arr: NDArray, axis: int, start: int, length: int) -> NDArray:
        return arr[(slice(None),) * axis + (slice(start, start + length, 1),)]
    
    def _scale_up_to(self, size: int):
        grow_to = int(2 ** math.ceil(math.log2(size + 1)))
        orig_shape = list(self._array.shape)
        orig_shape[self.grow_on] = grow_to
        
        new_storage = np.empty(orig_shape, dtype=self._array.dtype)
        self._numpy_narrow_equiv(
            new_storage, axis=self.grow_on, start=0, length=self.current_size
        )[:] = self._numpy_narrow_equiv(self._array, axis=self.grow_on, start=0, length=self.current_size)
        self._tensor = new_storage
        self._curr_max_size = grow_to
    
    @property
    def array(self) -> NDArray:
        return self._numpy_narrow_equiv(self._array, axis=self.grow_on, start=0, length=self.current_size)
    
    def __repr__(self) -> str:
        return f"AutoScalingArray(alloc={self._curr_max_size}, actual={self.current_size}, \n\tdata={self._array}\n)"
    
    def push(self, x: NDArray) -> None:
        data_size = x.shape[self.grow_on]
        if self.current_size + data_size >= self._curr_max_size:
            self._scale_up_to(self.current_size + data_size)
        assert (self.current_size + data_size) < self._curr_max_size
        
        self._numpy_narrow_equiv(
            self._array, axis=self.grow_on, start=0, length=data_size
        )[:] = x
        self.current_size += data_size
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        args = [data.array if isinstance(data, AutoScalingArray) else data for data in args]
        return getattr(ufunc, method)(*args, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        args = [data.array if isinstance(data, AutoScalingArray) else data for data in args]
        return func(*args, **kwargs)
