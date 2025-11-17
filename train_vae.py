from VAE import VAE
from dataloader import DataLoader
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import argparse


"""
some commands example commands to use:
# These two will test all functionalities for both datasets. Would not recommend changing epochs as training time will increase without much improvement to loss.
python train_vae.py --dset mnist_color --visualize_latent --generate_from_prior --generate_from_posterior
python train_vae.py --dset mnist_bw --visualize_latent --generate_from_prior --generate_from_posterior
"""
# Command-line arguments. Could add additional arguments for latent_dim, batch_size and number of generated images.
parser = argparse.ArgumentParser(description="Train a VAE on MNIST BW or Color.")
parser.add_argument('--dset', type=str, choices=['mnist_bw', 'mnist_color'], default='mnist_bw', help='Dataset to use for training')
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--visualize_latent", action="store_true", help="If set, visualize the latent space using t-SNE after training.")
parser.add_argument("--generate_from_prior", action="store_true", help="If set, generate samples from the prior p(z) after training.")
parser.add_argument("--generate_from_posterior", action="store_true", help="If set, generate samples from the posterior p(z|x) after training.")
args = parser.parse_args()



# Load data
my_data_loder = DataLoader(dset = args.dset, test=False)
tr_data = my_data_loder.load_data()
# Buffer_size set to 60,000 for shuffeling. Shuffles through entire dataset 
# prefetch speeds up the runtime of this file somewhat by providing parallelism. Does so by preparing next batch while computing current batch
train_ds = tr_data.shuffle(60000).batch(128).prefetch(tf.data.AUTOTUNE)

# Takes one example of the ds to get the input shape of the images: (28,28,1) or (28,28,3)
# Depending on the dataset
for x in train_ds.take(1):
    input_shape = x.shape[1:]
print(f"Input shape: {input_shape}")
channels = input_shape[-1]



# Initialize VAE model. Encoder and Decoder input_shape and channel_out depends on dataset.
encoder = Encoder(input_shape=input_shape)
decoder = Decoder(channel_out=channels)
model = VAE(encoder, decoder) 
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training loop
losses = []
epochs = args.epochs 
for epoch in range(epochs):  
    epoch_losses = []
    
    # For each batch, do a forward pass and collect the loss
    for batch in train_ds:
        loss = model.train(batch, optimizer)
        epoch_losses.append(loss.numpy())
    
    # Average over the losses in the epoch and collect for plotting
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()




### Visualization, generation and reconstructions

# Generate test data
test_loader = DataLoader(dset=args.dset, test=True)
test_ds = test_loader.load_data().batch(128)
# Number of samples to generate/reconstruct
num_generated = 10
# Fetch that number of samples from test set
test_batch = next(iter(test_ds))[:num_generated]

## Generate samples from prior or posterior
if args.generate_from_posterior:
    generated = model.generate_from_posterior(test_batch)
    fig, axes = plt.subplots(2, num_generated, figsize=(10, 6))
    axes[0, 0].set_title('Input')
    axes[1, 0].set_title('Reconstructed')
    for i in range(num_generated):
        # Plot input and reconstructed images
        axes[0, i].imshow(test_batch[i])
        axes[1, i].imshow(generated[i].numpy())
        # turn of axes so no ticks are shown
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()


            
    

if args.generate_from_prior:
    generated = model.generate_from_prior(n_samples=num_generated-1)
    fig, axes = plt.subplots(3,3, figsize=(10, 6))
    plt.suptitle("Samples Generated from Prior")
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(generated[i*3 + j].numpy())
            axes[i, j].axis('off')
    plt.show()

    



# Get latent representation for t-SNE visualization
if args.visualize_latent:
    # Number of samples to use for t-SNE
    num_samples = 2000
    # List to collect samples
    samples = []
    # Unbatch the dataset and iterativly collect the samples into a list
    for x in test_ds.unbatch().take(num_samples):
        samples.append(x.numpy())

    # np.stack to get array of shape (num_samples, 28, 28, channels) instead of list of arrays
    # This way I can passit to the model.
    samples = np.stack(samples)
    z = model.get_latent_representation(samples).numpy()

    # Define t-SNE model and transform latent representation
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    z_2d = tsne.fit_transform(z)

    # Plot t-SNE result. Could color by labels for clearer representation
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6)
    plt.title("Latent Space Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

