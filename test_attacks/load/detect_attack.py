import numpy as np
import torch
from scipy.special import logsumexp
from six.moves import xrange
import torch.nn as nn

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import sys
sys.path.append('/content/PGM-project')
sys.path.append('/content/PGM-project')
sys.path.append('../')
sys.path.append('../test_attacks')
sys.path.append('/home/boubacar/Documents/PGM project/PGM-project/test_attacks')
sys.path.append('./PGM-project/test_attacks')


from configs import * 
from test_attacks.load.load_classifier import load_classifier

# Assuming you have already imported torch and torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def compute_log_probabilities(logits, labels, description, compute_logit_distribution = False):
    log_prob_x = logsumexp(logits, axis=1)
    mean_log_prob_x = np.mean(log_prob_x)
    std_dev_log_prob_x = np.sqrt(np.var(log_prob_x))
    log_prob_x_given_y = torch.sum(logits * labels.cpu().numpy(), axis=1)
    mean_log_prob_x_given_y = []; std_dev_log_prob_x_given_y = []
    for i in xrange(labels.shape[1]):
        indices = np.where(labels[:, i] == 1)[0]
        mean_log_prob_x_given_y.append(np.mean(log_prob_x_given_y[indices]))
        std_dev_log_prob_x_given_y.append(np.sqrt(np.var(log_prob_x_given_y[indices])))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' \
          % (description, mean_log_prob_x, std_dev_log_prob_x, np.mean(mean_log_prob_x_given_y), np.mean(std_dev_log_prob_x_given_y)))
    
    results = [log_prob_x, mean_log_prob_x, std_dev_log_prob_x, log_prob_x_given_y, mean_log_prob_x_given_y, std_dev_log_prob_x_given_y]
    # compute distribution of the logits
    if compute_logit_distribution:
        mean_logits = []
        std_dev_logits = []
        mean_kl_divergence = []
        std_dev_kl_divergence = []
        mean_softmax = []
        for i in xrange(labels.shape[1]):
            indices = np.where(labels[:, i] == 1)[0]
            mean_logits.append(np.mean(logits[indices], 0))
            std_dev_logits.append(np.sqrt(np.var(logits[indices], 0)))

            logits_tmp = logits[indices] - logsumexp(logits[indices], axis=1)[:, np.newaxis]
            mean_softmax.append(np.mean(np.exp(logits_tmp), 0))
            kl_divergence = np.sum(mean_softmax[i] * (np.log(mean_softmax[i]) - logits_tmp), 1)
            
            mean_kl_divergence.append(np.mean(kl_divergence))
            std_dev_kl_divergence.append(np.sqrt(np.var(kl_divergence)))
        
        results.extend([mean_logits, std_dev_logits, mean_kl_divergence, std_dev_kl_divergence, mean_softmax]) 

    return results

def compute_detection_rate(x, mean_x, std_dev_x, alpha, plus):
    if plus:
        detection_rate = np.mean(x > mean_x + alpha * std_dev_x)
    else:
        detection_rate = np.mean(x < mean_x - alpha * std_dev_x)
    return detection_rate * 100

def find_optimal_alpha(x, mean_x, std_dev_x, target_rate = 5.0, plus = False):
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detection_rate = compute_detection_rate(x, mean_x, std_dev_x, alpha_now, plus)
    T = 0
    while np.abs(detection_rate - target_rate) > 0.01 and T < 20:
        if detection_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detection_rate = compute_detection_rate(x, mean_x, std_dev_x, alpha_now, plus)
        T += 1
    return alpha_now, detection_rate


def test_adversarial_attacks(dataset_name, model_name, attack_method, batch_size, is_targeted=False):
    if dataset_name == 'mnist':
        from utils import data_mnist
        train_images, train_labels, test_images, test_labels = data_mnist(train_start=0, train_end=200, test_start=0, test_end=30)
    
    num_samples, image_rows, image_cols = len(test_images), test_images.shape[0], test_images.shape[1]
    num_classes = 10
    
    model = load_classifier(model_name, dataset_name)
    model = model.to(device)
    if is_targeted:
        adversarial_inputs = np.array([instance for instance in test_images[:num_samples]], dtype=np.float32)
        one_hot_encoding = np.zeros((num_classes, num_classes), dtype=np.float32)
        one_hot_encoding[np.arange(num_classes), np.arange(num_classes)] = 1
        adversarial_inputs = adversarial_inputs.reshape((num_samples * num_classes, image_rows, image_cols, 1))
        adversarial_labels = np.array([one_hot_encoding] * num_samples, dtype=np.float32).reshape((num_samples * num_classes, num_classes))
    else:
        adversarial_inputs = test_images[:num_samples]
        adversarial_labels = test_labels[:num_samples]
    
    adversarial_inputs = adversarial_inputs.clone().detach().view(-1, 1, 28, 28).to(device)
    adversarial_labels = adversarial_labels.clone().detach().to(device)

    attack, attack_parameters = load_attack(attack_method)
    
    # Perform the attack
    adversarial_examples = []
    #one hot encoding of y
    labels = torch.zeros((num_samples, num_classes)).to(device)
    labels[np.arange(num_samples), adversarial_labels] = 1
    for i in range(0, num_samples, batch_size):
        adversarial_batch = attack(model, adversarial_inputs[i:min(num_samples,i+batch_size)], labels[i:min(num_samples,i+batch_size)], **attack_parameters)
        adversarial_examples.extend(adversarial_batch.cpu().detach().numpy())  # Move tensor to CPU before converting to numpy
    adversarial_examples = np.array(adversarial_examples, dtype=np.float32)
    print('-'*30)
    # Training the model on the train set and evaluating on test set with and without adversarial examples
    train_data_loader = DataLoader(list(zip(train_images.clone().detach().reshape(-1, 1, 28, 28).to(device), train_labels.clone().detach().to(device))), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train_model(train_data_loader, optimizer, num_epochs=1)

    
    # Evaluate the model on clean examples
    logits_clean = model.predict(torch.tensor(test_images[:num_samples].reshape(-1, 1, 28, 28)).to(device))
    _, predictions_clean = torch.max(logits_clean, 1)
    accuracy_clean = accuracy_score(test_labels[:num_samples], predictions_clean.cpu().numpy())  # Move tensor to CPU before converting to numpy
    print('Test accuracy on clean examples: {:.4f}'.format(accuracy_clean))

    # Evaluate the model on adversarial examples
    logits_adversarial = model.predict(torch.tensor(adversarial_examples.reshape(-1, 1, 28, 28)).to(device))
    _, predictions_adversarial = torch.max(logits_adversarial, 1)
    accuracy_adversarial = accuracy_score(test_labels[:num_samples], predictions_adversarial.cpu().numpy())  # Move tensor to CPU before converting to numpy
    print('Test accuracy on adversarial examples: {:.4f}'.format(accuracy_adversarial))

    correct_prediction_clean = torch.eq(predictions_clean, torch.tensor(test_labels[:num_samples]).to(device))
    accuracy_clean = torch.mean(correct_prediction_clean.float())
    success_rate_clean = 100 * (1 - accuracy_clean)
    successful_indices_clean = np.where(correct_prediction_clean.cpu().numpy() == 0)[0]
    
    correct_prediction_adversarial = torch.eq(predictions_adversarial, torch.tensor(adversarial_labels[:num_samples]).to(device))
    successful_indices_adversarial = np.where(correct_prediction_adversarial.cpu().numpy() == 0)[0]
    success_rate_adversarial = 100 * (1 - np.mean(correct_prediction_adversarial.cpu().numpy()))
    
    print("Attack success rate (clean/victim): %.2f%% / %.2f%%" % (success_rate_clean, success_rate_adversarial))

    # Compute the perturbation on successful attacks
    if len(successful_indices_clean) > 0:
        test_images_reshaped = test_images[successful_indices_clean].cpu().numpy().reshape(28,1,28,28)
        perturbations = adversarial_examples[successful_indices_clean] - test_images_reshaped

        l2_perturbations = np.sqrt(np.sum(perturbations ** 2, axis=(1, 2, 3)))
        linf_perturbations = np.max(np.abs(perturbations), axis=(1, 2, 3))
        l0_perturbations = np.sum(perturbations != 0, axis=(1, 2, 3))
        print("Perturbation for successful attack: L_2 = %.3f +- %.3f' % (np.mean(l2_perturbations), np.std(l2_perturbations)))")
        print("Perturbation for successful attack: L_inf = %.3f +- %.3f' % (np.mean(linf_perturbations), np.std(linf_perturbations)))")
        print("Perturbation for successful attack: L_0 = %.3f +- %.3f' % (np.mean(l0_perturbations), np.std(l0_perturbations)))")
        
        # Confidence of the attack (using entropy)
        log_prob_adv = logits_adversarial - logsumexp(logits_adversarial, axis=1)[:, np.newaxis]
        prob_adv = np.exp(log_prob_adv)
        print(log_prob_adv.mean(), prob_adv.mean())
        entropy = -torch.sum(prob_adv * log_prob_adv, axis=1)
        print("Confidence of the attack: %.3f +- %.3f" % (torch.mean(entropy), torch.std(entropy)))
    else:
        print("No successful attack")
        return None
    
    # compute logit on both clean and adversarial examples
    print('-'*30)
    logits_train = []
    print("Compute statistics on data")
    for i in xrange(int(train_images.shape[0] / batch_size)):
        batch_images = train_images[i*batch_size:(i+1)*batch_size]
        logits_train.append(model.predict(torch.tensor(batch_images.reshape(-1, 1, 28, 28)).to(device)).cpu().detach().numpy())
    logits_train = np.concatenate(logits_train)
    train_labels = train_labels[:logits_train.shape[0]]
    results_train = compute_log_probabilities(logits_train, train_labels, 'Train data', compute_logit_distribution=True)
    logits_clean = []
    for i in xrange(int(test_images.shape[0] / batch_size)):
        batch_images = test_images[i*batch_size:(i+1)*batch_size]
        logits_clean.append(model.predict(torch.tensor(batch_images.reshape(-1, 1, 28, 28)).to(device)).cpu().detach().numpy())
    logits_clean = np.concatenate(logits_clean, axis=0)
    
    # Now produce the logits
    results_clean = compute_log_probabilities(logits_clean, test_labels, 'Test data', compute_logit_distribution=True)
    results_adv = compute_log_probabilities(logits_adversarial, adversarial_labels, 'Adv data', compute_logit_distribution=True)
    log_prob_adv = logits_adversarial[successful_indices_clean] - logsumexp(logits_adversarial[successful_indices_clean], axis=1)[:, np.newaxis]
    prob_adv = np.exp(log_prob_adv)
    entropy = -np.sum(prob_adv * log_prob_adv, axis=1)
    print("Confidence of the attack: %.3f +- %.3f" % (torch.mean(entropy), torch.std(entropy)))
    
    # Use mean as reject threshold
    print('-'*30)
    results = {}
    results['success_rate_clean'] = success_rate_clean
    results['success_rate_adversarial'] = success_rate_adversarial
    results['mean_dist_l2'] = np.mean(l2_perturbations)
    results['std_dist_l2'] = np.std(l2_perturbations)
    results['mean_dist_linf'] = np.mean(linf_perturbations)
    results['std_dist_linf'] = np.std(linf_perturbations)
    results['mean_dist_l0'] = np.mean(l0_perturbations)
    results['std_dist_l0'] = np.std(l0_perturbations)
    results['mean_entropy'] = np.mean(entropy)
    results['std_entropy'] = np.std(entropy)
    
    alpha, detection_rate = find_optimal_alpha(results_train[0], results_train[1], results_train[2], plus=True)
    detection_rate = compute_detection_rate(results_train[0], results_train[1], results_train[2], alpha, plus=True)  
    delta_marginal = -(results_train[1] - alpha * results_train[2])
    print("Reject threshold (marginal): %.3f" % (delta_marginal))
    print("False alarm rate (reject < mean of logp(x) - %.2f * std): %.4f" % (alpha, detection_rate))
    results['FP_logpx'] = detection_rate
    detection_rate = compute_detection_rate(results_adv[0], results_train[1], results_train[2], alpha, plus=True)
    print("detection rate (reject < mean of logp(x) - %.2f * std): %.4f" % (alpha, detection_rate))
    results['TP_logpx'] = detection_rate

    
    false_positive_rate = []
    true_positive_rate = []
    delta_logit = []
    for i in xrange(num_classes):
        indices = np.where(adversarial_labels[:, i] == 1)[0]
        alpha, detection_rate = find_optimal_alpha(results_train[3][indices], results_train[4][i], results_train[5][i], plus=True)
        detection_rate = compute_detection_rate(results_train[3][indices], results_train[4][i], results_train[5][i], alpha, plus=True)
        false_positive_rate.append(detection_rate)
        delta_logit.append(-(results_train[4][i] - alpha * results_train[5][i]))
        
        indices = np.where(adversarial_labels[successful_indices_clean][:, i] == 1)[0]
        if len(indices) == 0:
            continue
        detection_rate = compute_detection_rate(results_adv[3][indices], results_train[4][i], results_train[5][i], alpha, plus=True)
        true_positive_rate.append(detection_rate)
    delta_logit = np.asarray(delta_logit, dtype=np.float32)
    print("Reject threshold (logit): %.3f +- %.3f" % (np.mean(delta_logit), np.std(delta_logit)))
    true_positive_rate = np.mean(true_positive_rate)
    false_positive_rate = np.mean(false_positive_rate)
    print("False alarm rate (reject < mean of logit - %.2f * std): %.4f" % (alpha, false_positive_rate))
    results['FP_logpxy'] = false_positive_rate
    print("detection rate (reject < mean of logit - %.2f * std): %.4f" % (alpha, true_positive_rate))
    results['TP_logpxy'] = true_positive_rate

    # KL detection
    logit_mean, _, kl_mean, kl_std, softmax_mean = results_train[-5:]
    false_positive_rate = []
    true_positive_rate = []
    delta_kl = []
    for i in xrange(num_classes):
        indices = np.where(train_labels[:, i] == 1)[0]
        logit_tmp = logits_train[indices] - logsumexp(logits_train[indices], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        alpha, detection_rate = find_optimal_alpha(kl, kl_mean[i], kl_std[i], plus=True)
        detection_rate = compute_detection_rate(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        false_positive_rate.append(detection_rate)
        delta_kl.append(kl_mean[i] + alpha * kl_std[i])

        indices = np.where(predictions_adversarial[successful_indices_clean][:, i] == 1)[0]
        if len(indices) == 0:	# no success attack, skip
            continue
        logit_tmp = logits_adversarial[indices] - logsumexp(logits_adversarial[indices], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        detection_rate = compute_detection_rate(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        true_positive_rate.append(detection_rate)
    delta_kl = np.asarray(delta_kl, dtype='f')
    print('delta_kl:', delta_kl)
    true_positive_rate = np.mean(true_positive_rate)
    false_positive_rate = np.mean(false_positive_rate)
    print('false alarm rate (reject > mean of conditional KL + %.2f * std): %.4f' % (alpha, false_positive_rate))
    results['FP_kl'] = false_positive_rate
    print('detection rate (reject > mean of conditional KL + %.2f * std): %.4f' % (alpha, true_positive_rate))
    results['TP_kl'] = true_positive_rate


if __name__ == '__main__':
    parser = ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--data', '-D', type=str, default='mnist')
    parser.add_argument('--conv', '-C', action='store_true', default=False)
    parser.add_argument('--guard', '-G', type=str, default='bayes_K10')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='fgsm')
    parser.add_argument('--victim', '-V', type=str, default='GBZ')
    parser.add_argument('--save', '-S', action='store_true', default=False)

    args = parser.parse_args()
    test_adversarial_attacks(args.data, args.victim, args.attack, args.batch_size, args.targeted)
