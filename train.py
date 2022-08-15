from dataloader import *
from Regularization import *
from model import *
from visualize_result import *
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_latent(autoencoder, data, num_batches=100):
    for i, batch in enumerate(data):
        x = batch["A"]
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1])
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def auto_kmean_train(autoencoder, Kmeans, train_dataloader, epochs1=100):
    opt1 = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    # criterion = nn.BCELoss()
    criterion1 = nn.MSELoss()
    start_time = time.time()
    lam = 2
    reconstruction_scores = []
    for n, epoch in enumerate(range(epochs1)):
        train_loss = 0.0
        labels = []
        for batch in train_dataloader:
            x = batch["A"] # 3 channels image
            x = x.to(device)  # GPU
            opt1.zero_grad()
            x_hat = autoencoder(x)
            loss = criterion1(x_hat, x) + lam * autoencoder.encoder.csd
            loss.backward()
            opt1.step()
            train_loss += loss.item()
            X = autoencoder.encoder(x)
            X = X.detach().cpu().numpy()
            Kmeans.fit(X)
            labels.append(Kmeans.labels_)
            if n == 10:
              reconstruction_scores.append(torch.mean(torch.mean(torch.mean(torch.squeeze((x - x_hat)**2), axis=1), axis=1), axis=1))
        train_loss = train_loss / len(train_dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f} \tElapsed time: {:.2f}s \tSCD: {:.5f}'.format(epoch, train_loss,
                                                                                               time.time() - start_time,
                                                                                               autoencoder.encoder.csd))

        if n%10 == 0:
            plot_latent(autoencoder, train_dataloader)
            imgs = next(iter(train_dataloader)) # batch-size
            real_A = imgs["A"].to(device)
            fake_A = autoencoder(real_A)
            # real_A: original image, fake_A: generated image
            img_sample = torch.cat((real_A.data, fake_A.data), -2)
            save_image(img_sample, f"result{n}.png", nrow=5, normalize=True)

    return autoencoder, Kmeans, labels, reconstruction_scores

if __name__ == '__main__':
    file_path = "./sam"
    train_transform1 = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
    train_dataset = ImageDataset(file_path, train_transform1, class_name="fish-big")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False
    )
    in_channels = 3
    out_channels = 3
    latent_space = 2
    autoencoder = Autoencoder(in_channels, out_channels, latent_space).to(device)  # GPU
    Kmean = KMeans(n_clusters=2)
    autoencoder, Kmeans, labels, reconstruction_scores = auto_kmean_train(autoencoder, Kmean,
                                                                            train_dataloader, epochs1=200)

    torch.save(autoencoder.state_dict(), "./autoencoder_best_model_100_2.pt")
    with open("./labels_100_2", "wb") as f:
        pickle.dump(labels, f)

    with open("./reconstruction_scores_100_2", "wb") as f:
        pickle.dump(reconstruction_scores, f)
    print("Model saved!")