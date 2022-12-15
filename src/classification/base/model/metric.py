import torch


def accuracy(output, target):
    """
    It takes in the output of the model and the target labels, and returns the accuracy of the model

    :param output: the output of the model
    :param target: the target variable
    :return: The accuracy of the model
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """
    > For each row in the output, we take the top k values and their indices, and then we check if the target value is in
    the top k indices

    :param output: the output of the model
    :param target: the target labels
    :param k: the number of top predictions to consider, defaults to 3 (optional)
    :return: The top k accuracy
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
