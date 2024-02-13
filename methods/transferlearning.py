import torch
from utils import DEVICE


def model_training(dataloader, model, loss_fn, lr, steps):
    n_epochs = int(steps / len(dataloader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = dataloader.batch_size
    for epoch in range(n_epochs):
        for input, label in dataloader:
            output = model(input)
            loss = loss_fn(output, label.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 50 == 0:
                print(f"Epochs: {epoch + 1}, loss: {loss.item():.5f}", end="\t\r")
            step += dataloader.batch_size

    return model


def create_pretrained_model(model, n_classes):
    num_ftrs = model.last.in_features
    model.last = torch.nn.Linear(num_ftrs, n_classes)
    return model
