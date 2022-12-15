import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.classification.base.utils.util import save_model, plot_class_distribution
from src.classification.hypertuned.model.hypermodel import ConvNet


def train_with_validation(config, num_epochs, learning_rate, weight_decay, train_loader, val_loader):
    """
    It trains the model, and returns the training and validation loss values

    :param config: the path to the config file
    :param num_epochs: The number of epochs to train for
    :param learning_rate: The learning rate for the optimizer
    :param weight_decay: The weight decay parameter is used to prevent overfitting
    :param train_loader: the training data loader
    :param val_loader: the validation set
    :return: The training and validation loss values are being returned.
    """

    model = ConvNet()
    print(model)

    # calculate weight for classes
    target = train_loader.dataset.dataset.dataset
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(target.targets),
                                   y=np.array(target.targets))
    weights = torch.tensor(weights, dtype=torch.float)

    plot_class_distribution(np.unique(target.targets, return_counts=True)[1], target.classes, config)

    # loss
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_accuracy = 0.0
    last_loss = 100
    patience = 2
    trigger_times = 0
    train_loss_values = []
    val_loss_values = []

    print("Begin training...")

    for epoch in range(1, num_epochs + 1):

        print(f'[Epoch {epoch}/{num_epochs}]')

        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0

        model.train()
        for data in tqdm(train_loader, desc="Train"):
            inputs, outputs = data

            optimizer.zero_grad()

            predicted = model(inputs)

            train_loss = criterion(predicted, outputs)

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item() * inputs.size(0)

        train_loss_value = running_train_loss / len(train_loader.dataset)
        train_loss_values.append(train_loss_value)

        with torch.no_grad():
            model.eval()
            for data in tqdm(val_loader, desc="Validate"):
                inputs, outputs = data

                predicted = model(inputs)
                val_loss = criterion(predicted, outputs)

                _, predicted = torch.max(predicted, 1)
                running_val_loss += val_loss.item() * inputs.size(0)
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        val_loss_value = running_val_loss / len(val_loader.dataset)
        val_loss_values.append(val_loss_value)

        accuracy = (100 * running_accuracy / total)

        if accuracy > best_accuracy:
            save_model(model, config)
            best_accuracy = accuracy

        print()
        print(f'Training Loss: {train_loss_value}')
        print(f'Validation Loss:{val_loss_value}')
        print(f'Accuracy on the validation set: {int(accuracy)}%')

        # Early stopping
        if val_loss_value > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return train_loss_values, val_loss_values

        else:
            trigger_times = 0

        last_loss = val_loss_value

    return train_loss_values, val_loss_values
