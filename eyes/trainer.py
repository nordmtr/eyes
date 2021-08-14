import numpy as np
import torch


def train(model, dataloader, optimizer, loss_fn, max_epochs=5, print_every_k_step=10):
    device = next(model.parameters()).device
    model.train()
    for epoch in range(max_epochs):
        running_loss = running_acc = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted.cpu() == labels).sum().item()
            acc = correct / total

            running_loss += loss.item()
            running_acc += acc
            if i % print_every_k_step == print_every_k_step - 1:
                print(
                    "[%d, %5d] loss: %.3f acc: %.3f"
                    % (epoch + 1, i + 1, running_loss / print_every_k_step, running_acc / print_every_k_step)
                )
                running_loss = running_acc = 0.0

    print("Finished Training")


def test(model, dataloader, loss_fn):
    device = next(model.parameters()).device
    model.eval()
    losses = []
    accs = []
    for data in dataloader:
        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = correct / total
        losses.append(loss.cpu().item())
        accs.append(accuracy)
    print("Loss: %.3f Acc: %.3f" % (np.mean(losses), np.mean(accs)))
