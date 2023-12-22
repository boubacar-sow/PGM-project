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


def visualize_attack(model_names, attack_method, models, data_loader, num_labels, num_epochs):
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

    images, labels = next(iter(data_loader))
    y = torch.zeros((len(labels), num_labels)).to(device)
    y[np.arange(len(labels)), labels] = 1
    images, labels = images.to(device), y.to(device)

    fig, axs = plt.subplots(ncols=len(models)+1, figsize=((len(models)+1)*10, 10))  # Create subplots in a single row

    # Display the original images
    grid_images = make_grid(images, nrow=8)  # Adjust nrow as needed
    np_grid_images = grid_images.cpu().numpy()
    np_grid_images = np.transpose(np_grid_images, (1, 2, 0))
    axs[0].imshow(np_grid_images, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    
    all_adv_images = []
    for i, (ax, model) in enumerate(zip(axs[1:], models)):  # Start from the second subplot
        # train the model before attacking
        X_train, Y_train, _, _ = data_mnist(train_start=0, train_end=7000, test_start=0, test_end=64)
        X_train = X_train / 255.0
        data_loader_train = DataLoader(list(zip(X_train.clone().detach().reshape(-1, 1, 28, 28).to(device), Y_train.clone().detach().to(device))), batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train_model(model_names[i], data_loader_train, optimizer, num_epochs=num_epochs)

        # Generate adversarial examples
        adv_images = attack(model, images, y, **attack_params)
        all_adv_images.append(adv_images)

    # Concatenate original and adversarial images
    for i, (ax, adv_images) in enumerate(zip(axs[1:], all_adv_images)):
        combined_images = torch.cat([images, adv_images])

        # Normalize the images to [0, 1] range
        combined_images = (combined_images - combined_images.min()) / (combined_images.max() - combined_images.min())

        # Create a grid of images
        grid_images = make_grid(combined_images, nrow=8)  # Adjust nrow as needed
        np_grid_images = grid_images.cpu().numpy()

        # Reshape and transpose the images for display
        np_grid_images = np.transpose(np_grid_images, (1, 2, 0))

        # Display the images
        ax.imshow(np_grid_images, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Model: {model_names[i]}')

    # Save the figure
    plt.savefig(f'{attack_method}_attack.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def main():
    parser = ArgumentParser(description='Visualize adversarial attacks.')
    parser.add_argument('--attack_method', type=str, default='fgsm')
    parser.add_argument('--model_names', nargs='+', default=['GBZ'])  # Add more model names as needed
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--num_labels', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    # Load your models here
    models = [load_classifier(model_name, args.data_name).to(device) for model_name in args.model_names]

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the test data
    testset = datasets.MNIST('./data', download=False, train=False, transform=transform)  
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    visualize_attack(args.model_names, args.attack_method, models, testloader, num_labels=args.num_labels, num_epochs=args.num_epochs)

if __name__ == '__main__':
    main()
