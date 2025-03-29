import time
import torch
from statistics import mean

from AutoScalingTensor import AutoScalingTensor, PagedScalingTensor

num_push  = 100000
push_size = 2000
data = [torch.arange(push_size) + (i * push_size) for i in range(num_push)]
all_time_A = []
all_time_B = []


for _ in range(50):
    time_A, time_B = [], []
    A = AutoScalingTensor(torch.empty(1000), grow_dim=0)
    B = PagedScalingTensor(torch.empty(1000), grow_dim=0, page_size=1_000_000)

    for i in range(num_push):
        startA = time.time()
        A.push(data[i])
        endA   = time.time()
        
        startB = time.time()
        B.push(data[i])
        endB   = time.time()
        
        time_A.append((endA - startA))
        time_B.append((endB - startB))
    
    all_time_A.append(time_A)
    all_time_B.append(time_B)

avg_time_A = [mean([all_time_A[i][j] for i in range(50)]) for j in range(num_push)]
avg_time_B = [mean([all_time_B[i][j] for i in range(50)]) for j in range(num_push)]


print("Average push AutoScalingTensor:", mean(avg_time_A))
print("Average push PagedScalingTensor:", mean(avg_time_B))

print("Max push AutoScalingTensor:", max(avg_time_A))
print("Max push PagedScalingTensor:", max(avg_time_B))
