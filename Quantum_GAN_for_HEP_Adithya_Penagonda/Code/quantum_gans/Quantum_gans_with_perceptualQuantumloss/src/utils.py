# quantum_gan_with_Perceptual_quantum_loss/src/utils.py

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pennylane as qml
import matplotlib.gridspec as gridspec

def perceptual_loss(real_images, fake_images, vgg, pca):
    # Inverse PCA transformation
    real_images = pca.inverse_transform(real_images.cpu().numpy())
    fake_images = pca.inverse_transform(fake_images.cpu().detach().numpy())

    # Reshape to image format
    real_images = torch.tensor(real_images, dtype=torch.float32)
    fake_images = torch.tensor(fake_images, dtype=torch.float32)
    
    real_images = real_images.view(-1, 1, 16, 16)
    fake_images = fake_images.view(-1, 1, 16, 16)
    
    real_images = real_images.to(real_images.device)
    fake_images = fake_images.to(fake_images.device)

    # Upsample to 224x224 and replicate channels
    upsample = transforms.Resize((224, 224))
    real_images = upsample(real_images)
    fake_images = upsample(fake_images)
    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)

    # Normalize images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    real_images = normalize(real_images)
    fake_images = normalize(fake_images)

    # Extract features
    with torch.no_grad():
        real_features = vgg(real_images)
        fake_features = vgg(fake_images)

    # Compute MSE loss
    loss = F.mse_loss(fake_features, real_features)
    return loss

def quantum_fidelity(real_data, fake_data, pca):
    n_qubits = real_data.size(1)
    dev_fidelity = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev_fidelity, interface='torch')
    def state_preparation(inputs):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        return qml.state()

    real_states = state_preparation(real_data).cpu().numpy()
    fake_states = state_preparation(fake_data).cpu().detach().numpy()

    fidelities = []
    for real_state, fake_state in zip(real_states, fake_states):
        fidelity = np.abs(np.vdot(real_state, fake_state)) ** 2
        fidelities.append(fidelity)
    return np.mean(fidelities)

def plot_losses(epoch, args,disc_loss,gen_loss):
    # Implement plotting of losses
    """
    Function for plotting loss of discriminator and generator
    
    Arguments: gen_loss(generator loss), disc_loss(discriminator loss), epochs(list of range=number of train steps)

    """
    print(gen_loss)
    print(disc_loss)
    # print(gen_loss.item())
    # print(disc_loss.item())

    # gen_loss_np = gen_loss.detach().numpy()  # Convert PyTorch tensor to NumPy array
    # disc_loss_np = disc_loss.detach().numpy()  # Convert PyTorch tensor to NumPy array
    #
    # gen_loss_np = gen_loss_np.reshape(-1, 1)
    # disc_loss_np = disc_loss_np.reshape(-1, 1)

    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    epochs = [i for i in range(epoch)]
    epoch = epochs[-1]
    # plot loss curve
    ax_loss = plt.subplot(gs[:,:4])
    ax_loss.set_xlim(0, 1.1*epoch)
    ax_loss.plot(epochs, gen_loss, label="Generator")
    ax_loss.plot(epochs, disc_loss, label="Discriminator")
    ax_loss.set_xlabel('Epoch', fontsize=20)
    ax_loss.set_ylabel('Loss', fontsize=20)
    ax_loss.grid(True)
    ax_loss.legend(fontsize=15)

    # Save the plot
    save_path = 'final_output\loss_plot.png'  # Default save path if not specified
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid memory issues

    print(f"Plot saved at {save_path}")

def save_models(generator, discriminator, epoch, args):
    torch.save(generator.state_dict(), f'outputs/models/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'outputs/models/discriminator_epoch_{epoch}.pth')
