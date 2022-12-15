import torch
from tqdm import tqdm


def test(model, test_loader):
    """
    It runs the model on the test set and prints the accuracy

    :param model: the model we're training
    :param test_loader: the test set
    """
    running_accuracy = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test"):
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print(f'Accuracy on the test set: {int(100 * running_accuracy / total)}%')
