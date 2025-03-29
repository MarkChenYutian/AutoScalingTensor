from abc import ABC, abstractmethod
import torch


class ScalingTensorBase(ABC):
    @abstractmethod
    def push(self, value: torch.Tensor) -> None: ...
    
    @abstractmethod
    def clear(self) -> None: ...
    
    @abstractmethod
    def __len__(self) -> int: ...
    
    @property
    @abstractmethod
    def data(self) -> torch.Tensor: ...
    
    @abstractmethod
    def __getitem__(self, index) -> torch.Tensor: ...
    
    @abstractmethod
    def __setitem__(self, index, value: torch.Tensor) -> None: ...


class AutoScalingTensor(ScalingTensorBase):
    def __init__(self, data: torch.Tensor, grow_dim: int = 0, curr_size: int = 0):
        self._data     = data
        self.grow_dim  = grow_dim
        self.curr_size = curr_size

    @property
    def data(self) -> torch.Tensor:
        return self._data.narrow(self.grow_dim, start=0, length=self.curr_size)
    
    @data.setter
    def data(self, value: torch.Tensor):
        self._data.narrow(self.grow_dim, start=0, length=self.curr_size).copy_(value)

    @staticmethod
    def get_scaling_factor(curr_capacity, incoming_size, factor=2):
        return max(curr_capacity * factor, curr_capacity + incoming_size)

    def __repr__(self) -> str:
        return f"AutoScalingTensor(data={self.data}, grow_on={self.grow_dim})"

    def push(self, value: torch.Tensor) -> None:
        data_size = value.size(self.grow_dim)

        # Expand the underlying tensor storage.
        if (self.curr_size + data_size) >= self._data.size(self.grow_dim):
            grow_target  = self.get_scaling_factor(self.curr_size, self.curr_size + data_size)
            shape_target = list(self._data.shape)
            shape_target[self.grow_dim] = grow_target
            
            new_storage  = torch.empty(shape_target, dtype=self._data.dtype, device=self._data.device)
            new_storage.narrow(dim=self.grow_dim, start=0, length=self.curr_size).copy_(
                self._data.narrow(dim=self.grow_dim, start=0, length=self.curr_size)
            )
            self._data = new_storage
        
        self._data.narrow(
            dim=self.grow_dim, start=self.curr_size, length=data_size
        ).copy_(value)
        self.curr_size += data_size

    def clear(self) -> None:
        self.curr_size = 0

    def __len__(self) -> int:
        return self.curr_size

    def __getitem__(self, index) -> torch.Tensor:
        elem_selector = tuple(
            [slice(None, None, None)] * self.grow_dim + [index]
        )
        return self.data.__getitem__(elem_selector)
    
    def __setitem__(self, index, value: torch.Tensor) -> None:
        elem_selector = tuple(
            [slice(None, None, None)] * self.grow_dim + [index]
        )
        return self.data.__setitem__(elem_selector, value)


class PagedScalingTensor(ScalingTensorBase):
    """Paged variant of the AutoScalingTensor, optimized for excessively large
    amount of data to bound the worst case time complexity of .push(...) to O(1).
    
    Designed to be interchangable with AutoScalingTensor, user should not see any
    difference in API usage, however:
    
    **WARNING** Unlike AutoScalingTensor which has .data property O(1) complexity, 
                the .data property of PagedScalingTensor has O(n) complexity.

    PagedScalingTensor = [torch.Tensor, ..., torch.Tensor, AutoScalingTensor]
                          ^^^^^^^^^^^^                          ^^^^^^^^^^^^
                          Fixed 'pages' that cannot scale       Head of the stack that receive new data.
    """
    def __init__(self, data: torch.Tensor, grow_dim: int = 0, curr_size: int = 0, page_size: int = 1_000_000):
        self.grow_dim     = grow_dim
        self.curr_size    = curr_size
        self.page_size    = page_size
        
        self.fixed_pages: list[torch.Tensor] = []
        num_pages = curr_size // page_size
        for page_idx in range(num_pages):
            self.fixed_pages.append(data.narrow_copy(dim=self.grow_dim, start=page_idx * page_size, length=page_size))
        
        self.scaling_head = AutoScalingTensor(
            data.narrow_copy(dim=self.grow_dim, start=num_pages * page_size, length=curr_size % page_size),
            grow_dim=grow_dim,
            curr_size=(curr_size % page_size)
        )
        
    @property
    def data(self) -> torch.Tensor:
        """Generate a torch.Tensor that contains all the data received by the PagedScalingTensor class.
        
        NOTE: This operation is very expensive as we need to concatenate all pages into a continuous tensor!
        """
        return torch.cat(self.fixed_pages + [self.scaling_head.data], dim=self.grow_dim)

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        """Set the value for the PagedScalingTensor 'as if' it is a contiguous tensor block.
        """
        value_size = value.size(self.grow_dim)
        num_pages  = value_size // self.page_size
        
        for page_index in range(num_pages):
            self.fixed_pages[page_index].copy_(value.narrow(
                dim    = self.grow_dim,
                start  = page_index * self.page_size,
                length = self.page_size
            ))
        
        self.scaling_head.data = value.narrow(
            dim   = self.grow_dim,
            start = num_pages * self.page_size,
            length= value_size - (num_pages * self.page_size)
        )

    def __repr__(self) -> str:
        return f"PagedScalingTensor(squashed={len(self.fixed_pages)}x{self.page_size} rows, recent_data={self.scaling_head.data})"

    def push(self, value: torch.Tensor) -> None:
        self.scaling_head.push(value)
        
        if len(self.scaling_head) > self.page_size:
            num_new_pages = len(self.scaling_head) // self.page_size
            for new_page in range(num_new_pages):
                self.fixed_pages.append(self.scaling_head.data.narrow(
                    dim=self.grow_dim, start=new_page * self.page_size, length=self.page_size
                ).clone())
            
            remain_values = self.scaling_head.data.narrow(
                dim=self.grow_dim,
                start=num_new_pages * self.page_size,
                length=len(self.scaling_head) - (num_new_pages * self.page_size)
            ).clone()
            self.scaling_head.clear()
            self.scaling_head.push(remain_values)
    
    def clear(self) -> None:
        self.scaling_head.clear()
        self.fixed_pages = []

    def __len__(self) -> int:
        return (len(self.fixed_pages) * self.page_size) + len(self.scaling_head)

    def __getitem__(self, index: int | torch.Tensor) -> torch.Tensor:
        """Retrieve items from the PagedScalingTensor.

        Args:
            index (int | torch.Tensor):
            Can be any one of the following
            *   an integer value
            *   a one-dimension torch.Tensor with dtype int64
            *   a one-dimension torch.Tensor with dtype bool as masking

        Returns:
            torch.Tensor: retrieved rows, if more than one index value is provided (through mask or tensor indexing),
            the ordering of retrieved values match the ordering of the index.
        """
        if isinstance(index, int):
            page_index = index // self.page_size
            elem_index = index % self.page_size
            if page_index > len(self.fixed_pages):
                return self.scaling_head._data \
                    .narrow(dim=self.grow_dim, start=elem_index, length=1) \
                    .squeeze(dim=self.grow_dim)
            else:
                return self.fixed_pages[page_index] \
                    .narrow(dim=self.grow_dim, start=elem_index, length=1) \
                    .squeeze(dim=self.grow_dim)
        
        elif isinstance(index, torch.Tensor) and (index.ndim == 1) and (index.dtype == torch.long):
            # Initialize a storage tensor to collect all results
            orig_shape = list(self.scaling_head.data.shape)
            orig_shape[self.grow_dim] = index.size(0)
            result_tensor = torch.empty(orig_shape, dtype=self.scaling_head.data.dtype, device=self.scaling_head.data.device)
            
            # Retrieve results
            page_index = index // self.page_size
            elem_index = index % self.page_size
            
            pages = page_index.unique(sorted=False)
            for page in pages:
                page = page.item()
                page_mask = page_index == page
                elem_selector = tuple(
                    [slice(None, None, None)] * self.grow_dim + [elem_index[page_mask]]
                )
                if page >= len(self.fixed_pages):
                    elems = self.scaling_head.data.__getitem__(elem_selector)
                else:
                    elems = self.fixed_pages[page].__getitem__(elem_selector)
                result_tensor[page_mask] = elems
            
            return result_tensor
        
        elif isinstance(index, torch.Tensor) and (index.ndim == 1) and (index.dtype == torch.bool):
            assert index.size(0) == len(self)
            
            # Initialize a storage tensor to collect all results
            orig_shape = list(self.scaling_head.data.shape)
            orig_shape[self.grow_dim] = int(index.sum().item())
            result_tensor = torch.empty(orig_shape, dtype=self.scaling_head.data.dtype, device=self.scaling_head.data.device)
            
            # Retrieve Results
            current_frontier = 0
            for page_idx in range(len(self.fixed_pages)):
                page_index = index[page_idx * self.page_size : (page_idx + 1) * self.page_size]
                if not torch.any(page_index): continue
                
                elem_selector = tuple(
                    [slice(None, None, None)] * self.grow_dim + [page_index]
                )
                elems = self.fixed_pages[page_idx].__getitem__(elem_selector)
                result_tensor.narrow(
                    dim=self.grow_dim, start=current_frontier, length=elems.size(self.grow_dim)
                ).copy_(elems)
                
                current_frontier += elems.size(self.grow_dim)
            
            page_index = index[len(self.fixed_pages) * self.page_size :]
            if not torch.any(page_index): return result_tensor
            
            elem_selector = tuple(
                [slice(None, None, None)] * self.grow_dim + [page_index]
            )
            elems = self.scaling_head.data.__getitem__(elem_selector)
            result_tensor.narrow(
                dim=self.grow_dim, start=current_frontier, length=elems.size(self.grow_dim)
            ).copy_(elems)
            return result_tensor
            
        else:
            raise NotImplementedError(f"Did not implemented a __getitem__ method for your input.")
    
    def __setitem__(self, index, value: torch.Tensor) -> None:
        if isinstance(index, int):
            page_index = index // self.page_size
            elem_index = index % self.page_size
            if page_index > len(self.fixed_pages):
                self.scaling_head._data \
                    .narrow(dim=self.grow_dim, start=elem_index, length=1) \
                    .copy_(value.unsqueeze(self.grow_dim))
            else:
                self.fixed_pages[page_index] \
                    .narrow(dim=self.grow_dim, start=elem_index, length=1) \
                    .copy_(value.unsqueeze(self.grow_dim))
        
        elif isinstance(index, torch.Tensor) and (index.ndim == 1) and (index.dtype == torch.long):
            # Retrieve results
            page_index = index // self.page_size
            elem_index = index % self.page_size
            
            pages = page_index.unique(sorted=False)
            for page in pages:
                page = page.item()
                page_mask = page_index == page
                dst_selector = tuple(
                    [slice(None, None, None)] * self.grow_dim + [elem_index[page_mask]]
                )
                src_selector = tuple(
                    [slice(None, None, None)] * self.grow_dim + [page_mask]
                )
                
                if page >= len(self.fixed_pages):
                    self.scaling_head.__setitem__(elem_index[page_mask], value.__getitem__(src_selector))
                else:
                    self.fixed_pages[page].__setitem__(dst_selector, value.__getitem__(src_selector))
        
        elif isinstance(index, torch.Tensor) and (index.ndim == 1) and (index.dtype == torch.bool):
            assert index.size(0) == len(self)
            
            # Initialize a storage tensor to collect all results
            orig_shape = list(self.scaling_head.data.shape)
            orig_shape[self.grow_dim] = int(index.sum().item())
            
            # Retrieve Results
            current_frontier = 0
            for page_idx in range(len(self.fixed_pages)):
                page_index = index[page_idx * self.page_size : (page_idx + 1) * self.page_size]
                num_elem = int(page_index.sum())
                if num_elem == 0: continue
                
                elem_selector = tuple(
                    [slice(None, None, None)] * self.grow_dim + [page_index]
                )
                self.fixed_pages[page_idx].__setitem__(elem_selector, value.narrow(dim=self.grow_dim, start=current_frontier, length=num_elem))
                current_frontier += num_elem
            
            scaling_head_index = index[len(self.fixed_pages) * self.page_size :]
            num_elem = int(scaling_head_index.sum())
            if num_elem == 0: return
            
            self.scaling_head.__setitem__(scaling_head_index, value.narrow(dim=self.grow_dim, start=current_frontier, length=num_elem))
            
        else:
            raise NotImplementedError(f"Did not implemented a __getitem__ method for your input.")
