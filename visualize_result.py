from dataloader import *
from model import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def imshow1(img):
    img = img / 2 + 0.5
    plt.imshow(img[0])
def imshow2(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

def result(autoencoder, labels, train_dataloader):
    for i, batch in enumerate(train_dataloader):
        x = batch["A"]
        y = batch["A"]
        output = autoencoder(x.to(device))
        images = y.numpy()
        output = output.view(10, 3, 256, 256)
        output = output.detach().cpu().numpy()
        fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(12,4))
        for idx in np.arange(20):
            ax1 = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            if idx < 10:
              imshow1(output[idx])
              ax1.set_title(labels[i][idx])
            else:
              imshow2(images[idx-10])
              ax1.set_title(labels[i][idx-10])
        plt.show()

        if i == 18:
            break

if __name__ == '__main__':
    file_path = "./sam"
    train_dataset = ImageDataset(file_path, train_transform1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False
    )
    in_channels = 3
    out_channels = 3
    latent_space = 2
    autoencoder = Autoencoder(in_channels, out_channels, latent_space).to(device)  # GPU
    autoencoder.load_state_dict(torch.load("./autoencoder_best_model_100_2.pt"))
    with open("./labels_100_2", 'rb') as f:
        labels = pickle.load(f)
    with open("./reconstruction_scores_100_2", 'rb') as f:
        reconstruction_scores = pickle.load(f)

    reconstruction_scores1 = []
    labels1 = []
    for i, w in enumerate(reconstruction_scores):
        r = w.cpu().detach().numpy()
        for ii, n in enumerate(r):
            reconstruction_scores1.append(n)
            labels1.append(labels[i][ii])

    anomaly_data = pd.DataFrame({'recon_score': reconstruction_scores1})

    # if our reconstruction scores our normally distributed we can use their statistics
    anomaly_data.describe()
    # plotting the density will give us an idea of how the reconstruction scores are distributed
    plt.xlabel('Reconstruction Score')
    anomaly_data['recon_score'].plot.hist(bins=10)
    # Or we assume our reconstructions are normally distributed and label anomalies as those
    # that are a number of standard deviations away from the mean
    recon_mean = np.mean(reconstruction_scores1)
    recon_stddev = np.std(reconstruction_scores1)

    stats_threshold = recon_mean + recon_stddev - 0.019
    print(stats_threshold)

    important_frames1 = []
    all_detected_frames1 = []
    for i, score in enumerate(reconstruction_scores1):
        if labels1[i] == 0:
            all_detected_frames1.append(i + 1)
            if score > stats_threshold:
                print(i + 1, score)
                img_Z = Image.open(
                    f"./sam/Images/fish-big/frame{i + 1}.jpg")
                plt.imshow(img_Z)
                important_frames1.append(i + 1)
            else:
                print(i + 1, "x", score)
                img_Z = Image.open(
                    f"./sam/Images/fish-big/frame{i + 1}.jpg")
                plt.imshow(img_Z)
            plt.show()
    with open("./important_frames", "wb") as f:
        pickle.dump(important_frames1, f)

    predict_frames = np.ones(199)
    original_frames = np.ones(199)
    len(predict_frames)
    original = [149, 175, 161, 160, 148, 162, 48, 176, 177, 163, 49, 188, 198, 173, 172, 166, 199, 170, 164, 158, 159,
                165, 171, 14, 28, 29, 15, 17, 16, 3, 7, 12, 6, 39, 11, 10, 38, 21, 8, 9, 34, 20, 36, 22, 23, 37, 33, 27,
                26, 32, 18, 30, 31, 25, 19, 197, 183, 168, 42, 56, 154, 155, 169, 43, 182, 196, 194, 55, 41, 157, 40,
                195, 181, 185, 191, 146, 152, 50, 44, 45, 51, 153, 147, 190, 184, 192, 186, 151, 47, 179, 53, 52, 144,
                150, 187, 174, 4, 5, 180, 54, 193]
    original = natsorted(original)
    for org in original:
        original_frames[org - 1] = 0
    for evt in important_frames1:
        predict_frames[evt - 1] = 0
    print(original_frames)
    print(predict_frames)
    print(accuracy_score(original_frames, predict_frames))
    # Get the confusion matrix
    cf_matrix = confusion_matrix(original_frames, predict_frames)
    sns.heatmap(cf_matrix, annot=True)