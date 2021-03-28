import torch
from utills import move_to
from tqdm import tqdm


def train_simple_network(
    model,
    loss_func,
    training_loader,
    epochs=20,
    device="cpu"
):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.to(device)
    for _ in tqdm(range(epochs), desc="Epoch"):
        model = model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(training_loader, desc="Batch", leave=False):
            inputs = move_to(inputs, device)
            labels = move_to(labels, device)

            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_func(y_hat, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
