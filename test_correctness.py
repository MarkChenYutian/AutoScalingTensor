import pytest
import random
import torch

from AutoScalingTensor import PagedScalingTensor


def test_long_indexing_tensor():    
    Test = PagedScalingTensor(torch.zeros(10, 3, dtype=torch.long), grow_dim=0, curr_size=0, page_size=100)
    Ref  = torch.arange(1000).unsqueeze(-1).repeat(1, 3)
    Test.push(Ref)

    for i in range(100):
        long_selector = torch.tensor(random.choices([i for i in range(1000)], k=200)).long()
        assert torch.allclose(Test[long_selector], Ref[long_selector])


def test_bool_indexing_tensor():    
    Test = PagedScalingTensor(torch.zeros(10, 3, dtype=torch.long), grow_dim=0, curr_size=0, page_size=100)
    Ref  = torch.arange(1000).unsqueeze(-1).repeat(1, 3)
    Test.push(Ref)

    for i in range(100):
        long_selector = torch.tensor(random.choices([i for i in range(1000)], k=200)).long()
        bool_selector = torch.zeros(1000, dtype=torch.bool)
        bool_selector[long_selector] = True
        
        assert torch.allclose(Test[bool_selector], Ref[bool_selector])
        print(f"Pass bool selector test")

def test_long_set_tensor():    
    Test = PagedScalingTensor(torch.zeros(10, 3, dtype=torch.long), grow_dim=0, curr_size=0, page_size=100)
    Ref  = torch.arange(1000).unsqueeze(-1).repeat(1, 3)
    Test.push(Ref)

    for i in range(100):
        long_selector = torch.tensor(list(set(random.choices([i for i in range(1000)], k=200)))).long()
        value_tensors = (torch.randn((long_selector.size(0), 3)) * 1000).long()
        
        Test[long_selector] = value_tensors
        Ref[long_selector]  = value_tensors
        
        assert torch.allclose(Test.data, Ref)


def test_bool_set_tensor():
    Test = PagedScalingTensor(torch.zeros(10, 3, dtype=torch.long), grow_dim=0, curr_size=0, page_size=100)
    Ref  = torch.arange(1000).unsqueeze(-1).repeat(1, 3)
    Test.push(Ref)

    for i in range(100):
        long_selector = torch.tensor(random.choices([i for i in range(1000)], k=200)).long()
        bool_selector = torch.zeros(1000, dtype=torch.bool)
        bool_selector[long_selector] = True
        value_tensors = (torch.zeros((int(bool_selector.sum()), 3)) * 1000).long()
        
        Test[bool_selector] = value_tensors
        Ref[bool_selector] = value_tensors
        assert torch.allclose(Test.data, Ref)

test_long_set_tensor()
