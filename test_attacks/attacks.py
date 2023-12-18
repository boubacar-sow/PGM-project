import numpy as np
import torch
import torch.nn as nn

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from configs import * 
from load_classifier import load_classifier


def test_attacks(data_name, model_name, attack_method, eps, batch_size, targeted=False):
    
    
    if data_name == 'mnist':
        from project.utils import data_mnist
        X_train, Y_train, X_test, Y_test = data_mnist()
    
    source_samples, img_rows, img_cols, img_channels = X_test.shape
    nb_classes = Y_test.shape[1]
    
    model = load_classifier(model_name, data_name)
    
    if targeted:
        adv_inputs = np.array([instance for instance in X_test[:source_samples]], dtype=np.float32)
        one_hot = np.zeros((nb_classes, nb_classes), dtype=np.float32)
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1
        adv_inputs = adv_inputs.reshape((source_samples * nb_classes, img_rows, img_cols, 1))
        adv_ys = np.array([one_hot] * source_samples, dtype=np.float32).reshape((source_samples * nb_classes, nb_classes))
    else:
        adv_inputs = X_test[:source_samples]
        adv_ys = Y_test[:source_samples]
        
    # Perform the attack
    adv_examples = []
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    for i in range(source_samples):
        input_var = torch.autograd.Variable(adv_inputs[i:i+1], requires_grad=True)
        target_var = torch.autograd.Variable(adv_ys[i:i+1])

        # Forward pass
        output = model(input_var)
        loss = criterion(output, target_var)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Apply the attack
        input_var.data = input_var.data + ((eps * input_var.grad.data.sign()))
        adv_examples.append(input_var.data.numpy())

    adv_examples = np.concatenate(adv_examples, axis=0)

    # Evaluate the model on adversarial examples
    adv_labels_pred = model.predict(adv_examples)
    adv_accuracy = accuracy_score(adv_ys, adv_labels_pred)
    print('Adversarial accuracy: {:.2f}%'.format(adv_accuracy * 100))

    return adv_examples

def main():
    parser = ArgumentParser(description='Run adversarial attacks.')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--attack_method', type=str, default='fgsm')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--targeted', action='store_true', default=False)
    args = parser.parse_args()

    test_attacks(args.data_name, args.model_name, args.attack_method, args.eps, args.batch_size, args.targeted)

if __name__ == '__main__':
    main()