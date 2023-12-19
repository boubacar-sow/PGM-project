import numpy as np
import torch
import torch.nn as nn

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from configs import * 
from test_attacks.load.load_classifier import load_classifier

def test_attacks(data_name, model_name, attack_method, eps, batch_size, targeted=False):
    
    
    if data_name == 'mnist':
        from utils import data_mnist
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=30, test_start=0, test_end=10)
    
    source_samples, img_rows, img_cols = len(X_test), X_test.shape[0], X_test.shape[1]
    nb_classes = 10
    
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
    
    adv_inputs = adv_inputs.view(-1, 1, 28, 28)

    attack, attack_params = load_attack(attack_method)
    # Perform the attack
    adv_examples = []
    #one hot encoding of y
    y = torch.zeros((source_samples, nb_classes))
    y[np.arange(source_samples), adv_ys] = 1
    for i in range(0, source_samples, batch_size):
        adv_batch = attack(model, adv_inputs[i:min(source_samples,i+batch_size)], y[i:min(source_samples,i+batch_size)], **attack_params)
        adv_examples.extend(adv_batch)
    adv_examples = np.array([adv_example.detach().numpy() for adv_example in adv_examples], dtype=np.float32)
    print('-'*30)

   # Evaluate the model on adversarial examples
    p_y_pred = model.predict(torch.tensor(adv_examples.reshape(-1, 1, 28, 28)))
    _, y_preds = torch.max(p_y_pred, 1)
    print(y_preds)
    print(Y_test[:source_samples])
    accuracy = accuracy_score(Y_test[:source_samples], y_preds)
    print('Test accuracy on adversarial examples: {:.4f}'.format(accuracy))

    
    return adv_examples

def main():
    parser = ArgumentParser(description='Run adversarial attacks.')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='GBZ')
    parser.add_argument('--attack_method', type=str, default='fgsm')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--targeted', action='store_true', default=False)
    args = parser.parse_args()

    test_attacks(args.data_name, args.model_name, args.attack_method, args.eps, args.batch_size, args.targeted)

if __name__ == '__main__':
    main()