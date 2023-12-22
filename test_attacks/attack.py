import numpy as np
import torch
import torch.nn as nn

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import sys

sys.path.append("/home/cmap/Desktop/MVA/PGM")
# print(sys.path)
from configs import *
from test_attacks.load.load_classifier import load_classifier

# Assuming you have already imported torch and torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_attacks(data_name, model_name, attack_method, eps, batch_size, targeted=False):
    print("Training the model")
    if data_name == "mnist":
        from utils import data_mnist

        X_train, Y_train, X_test, Y_test = data_mnist(
            train_start=0, train_end=25000, test_start=0, test_end=64
        )
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    source_samples, img_rows, img_cols = len(X_test), X_test.shape[0], X_test.shape[1]
    nb_classes = 10
    # Training the model on the train set and evaluating on test set with and without adversarial examples
    data_loader_train = DataLoader(
        list(
            zip(
                X_train.clone().detach().reshape(-1, 1, 28, 28).to(device),
                Y_train.clone().detach().to(device),
            )
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model = load_classifier(model_name, data_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train_model(model_name, data_loader_train, optimizer, num_epochs=1)

    # Evaluate the model on clean examples
    p_y_pred = model.predict(
        model_name,
        torch.tensor(X_test[:source_samples].reshape(-1, 1, 28, 28)).to(device),
    )
    _, y_preds = torch.max(p_y_pred, 1)
    accuracy = accuracy_score(
        Y_test[:source_samples], y_preds.cpu().numpy()
    )  # Move tensor to CPU before converting to numpy
    print("Test accuracy on clean examples: {:.4f}".format(accuracy))
    

    print("Testing Attacks on Model: ", model_name)
    if targeted:
        adv_inputs = np.array(
            [instance for instance in X_test[:source_samples]], dtype=np.float32
        )
        one_hot = np.zeros((nb_classes, nb_classes), dtype=np.float32)
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1
        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, 1)
        )
        adv_ys = np.array([one_hot] * source_samples, dtype=np.float32).reshape(
            (source_samples * nb_classes, nb_classes)
        )
    else:
        adv_inputs = X_test[:source_samples]
        adv_ys = Y_test[:source_samples]

    adv_inputs = adv_inputs.clone().detach().view(-1, 1, 28, 28).to(device)
    adv_ys = adv_ys.clone().detach().to(device)

    attack, attack_params = load_attack(attack_method)

    # Perform the attack
    adv_examples = []
    # one hot encoding of y
    y = torch.zeros((source_samples, nb_classes)).to(device)
    y[np.arange(source_samples), adv_ys] = 1
    for i in range(0, source_samples, batch_size):
        adv_batch = attack(
            model,
            adv_inputs[i : min(source_samples, i + batch_size)],
            y[i : min(source_samples, i + batch_size)],
            **attack_params
        )
        adv_examples.extend(
            adv_batch.cpu().detach().numpy()
        )  # Move tensor to CPU before converting to numpy
    adv_examples = np.array(adv_examples, dtype=np.float32)
    print("-" * 30)



    # Evaluate the model on adversarial examples
    p_y_pred = model.predict(
        model_name, torch.tensor(adv_examples.reshape(-1, 1, 28, 28)).to(device)
    )
    _, y_preds = torch.max(p_y_pred, 1)
    accuracy = accuracy_score(
        Y_test[:source_samples], y_preds.cpu().numpy()
    )  # Move tensor to CPU before converting to numpy
    print("Test accuracy on adversarial examples: {:.4f}".format(accuracy))

    return adv_examples


def main():
    parser = ArgumentParser(description="Run adversarial attacks.")
    parser.add_argument("--data_name", type=str, default="mnist")
    parser.add_argument("--model_name", type=str, default="GBZ")
    parser.add_argument("--attack_method", type=str, default="fgsm")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--targeted", action="store_true", default=False)
    args = parser.parse_args()

    test_attacks(
        args.data_name,
        args.model_name,
        args.attack_method,
        args.eps,
        args.batch_size,
        args.targeted,
    )


if __name__ == "__main__":
    main()
