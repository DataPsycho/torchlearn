import torch


def f(x):
    return torch.pow((x - 2.0), 2)


arbitrary_value = torch.tensor([-3.5], requires_grad=True)
grad_value = f(arbitrary_value)
grad_value.backward()
print(arbitrary_value.grad)

# Algorithmic Gradient Descend (Version 1 Manual)
init_x = torch.tensor([-3.5], requires_grad=True)
x_cur = init_x.clone()
x_prev = x_cur * 100
epsilon = 1e-5
eta = 0.1

while torch.norm(x_cur - x_prev) > epsilon:
    x_prev = x_cur.clone()
    value = f(init_x)
    value.backward()
    init_x.data -= eta * init_x.grad
    init_x.grad.zero_()
    x_cur = init_x.data

print(x_cur)

# Estimate using parameter (Version 2)
x_param = torch.nn.Parameter(torch.tensor([-3.5]), requires_grad=True)
optimizer = torch.optim.SGD([x_param], lr=eta)

for epoch in range(60):
    optimizer.zero_grad()  # x.grad.zero_()
    loss_incurred = f(x_param)
    loss_incurred.backward()
    optimizer.step()  # x.data -= eta * x.grad
    print(x_param.data)