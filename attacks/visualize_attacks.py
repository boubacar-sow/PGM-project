import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision.utils import make_grid
from torchvision import datasets, transforms

sys.path.append('/content/PGM-project')
sys.path.append('../')
sys.path.append('../test_attacks')
sys.path.append('/home/boubacar/Documents/PGM project/PGM-project/test_attacks')
sys.path.append('./PGM-project/test_attacks')

# Import the attack methods
from attacks.fast_gradient_sign_method import fast_gradient_sign_method
from attacks.noise import noise
from attacks.momentum_iterative_method import momentum_iterative_method
from attacks.projected_gradient_descent import projected_gradient_descent
from attacks.carlini_wagner_l2 import carlini_wagner_l2
from attacks.spsa import spsa

from configs import * 
from test_attacks.load.load_classifier import load_classifier
from utils import data_mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_attack(attack_method, model, data_loader, num_labels):
    # Load the attack method
    attack_methods = {
        'fgsm': fast_gradient_sign_method,
        'noise': noise,
        'momentum_iterative_method': momentum_iterative_method,
        'projected_gradient_descent': projected_gradient_descent,
        'carlini_wagner_l2': carlini_wagner_l2,
        'spsa': spsa
    }
    attack, attack_params = load_attack(attack_method)
    
    # train the model before attacking
    # Build X_train, Y_train, X_test, Y_test
    X_train, Y_train, _, _ = data_mnist(train_start=0, train_end=7000, test_start=0, test_end=64)
    X_train = X_train / 255.0
    data_loader_train = DataLoader(list(zip(X_train.clone().detach().reshape(-1, 1, 28, 28).to(device), Y_train.clone().detach().to(device))), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train_model(data_loader_train, optimizer, num_epochs=5)

    # Get a batch of data
    images, labels = next(iter(data_loader))
    # one hot encoding of the labels
    y = torch.zeros((len(labels), num_labels)).to(device)
    y[np.arange(len(labels)), labels] = 1
    
    images, labels = images.to(device), y.to(device)

    # Generate adversarial examples
    adv_images = attack(model, images, y, **attack_params)

    # Generate adversarial examples
    adv_images = attack(model, images, y, **attack_params)
    # Normalize the images to [0, 1] range
    images = (images - images.min()) / (images.max() - images.min())
    adv_images = (adv_images - adv_images.min()) / (adv_images.max() - adv_images.min())

    # Create a grid of images
    grid_images = make_grid(torch.cat([images, adv_images]), nrow=8)  # Adjust nrow as needed
    np_grid_images = grid_images.cpu().numpy()

    # Reshape and transpose the images for display
    np_grid_images = np.transpose(np_grid_images, (1, 2, 0))

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the images
    ax.imshow(np_grid_images, cmap='gray')
    ax.axis('off')

    # Save the figure
    plt.savefig(f'{attack_method}_attack.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def main():
    parser = ArgumentParser(description='Visualize adversarial attacks.')
    parser.add_argument('--attack_method', type=str, default='fgsm')
    parser.add_argument('--model_name', type=str, default='GBZ')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--num_labels', type=int, default=10)
    args = parser.parse_args()

    # Load your model here
    model = load_classifier(args.model_name, args.data_name)
    model = model.to(device)
 
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x / 255)])

# Download and load the test data
    testset = datasets.MNIST('./data', download=False, train=False, transform=transform)  
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    visualize_attack(args.attack_method, model, testloader, num_labels=args.num_labels)

if __name__ == '__main__':
    main()
