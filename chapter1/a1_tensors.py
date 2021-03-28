import torch
import timeit
from utills import move_to

# Tensors
torch_scalar = torch.tensor(3.14)
torch_vector = torch.tensor([1, 2, 3, 4])
torch_matrix = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
])
torch_tensor3d = torch.tensor([
     [
         [1, 2, 3],
         [4, 5, 6],
     ],
     [
         [7, 8, 9],
         [10, 11, 12],
     ],
     [
         [13, 14, 15],
         [16, 17, 18],
     ],
     [
         [19, 20, 21],
         [22, 23, 24],
     ]
])

# check cpu time
x = torch.rand(2**11, 2**11)

time_cpu = timeit.timeit("x@x", globals=globals(), number=100)

# Check for CUDA Availability
print("Is CUDA available? :", torch.cuda.is_available())
device = torch.device("cuda")

x = x.to(device)
time_gpu = timeit.timeit("x@x", globals=globals(), number=100)

x = torch.rand(128, 128).to(device)
y = torch.rand(128, 128)
x*y

some_tensors = [torch.tensor(1), torch.tensor(2)]
print(some_tensors)
print(move_to(some_tensors, device))