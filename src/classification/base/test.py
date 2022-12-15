import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.classification.base.utils.util import load_model


def test(dir_config, test_loader):
    """
    It loads the model, then runs it on the test set, and prints out the accuracy and confusion matrix

    :param dir_config: the directory where the model is saved
    :param test_loader: the test data loader
    """

    model = load_model(dir_config)

    running_accuracy = 0
    total = 0
    model.eval()

    nb_classes = 10
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test"):
            inputs, outputs = data

            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            y_pred_list.append(predicted.numpy())
            y_test_list.append(outputs.numpy())

            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

            for t, p in zip(outputs.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        print(f'Accuracy on the test set: {int(100 * running_accuracy / total)}%')
        print(confusion_matrix.diag() / confusion_matrix.sum(1))

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_test_list = [a.squeeze().tolist() for a in y_test_list]

        ml = MultiLabelBinarizer().fit(y_test_list)

        print(classification_report(ml.transform(y_test_list),
                                    ml.transform(y_pred_list)))
