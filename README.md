# Video_summarization_project
Unsupervised video summarization using ITL- Autoencoder

In a video, there are so many frames that are not important to see or check contents. These unimportant frames make us waste the time. We can solve this problem by detecting important objects in a video and making video time shorter automatically. Detecting objects is performed well in computer vision. However, this good performance is for not only important objects but also unimportant objects in the video. Detecting only important objects is a challenging problem in computer vision. If we can detect only important objects in a video, there will be many applications we can apply in various fields. For example, in underwater circumstances, we can check what is happening by recording a video. However, it is difficult to sort out which parts are important and unimportant in a video. Moreover, it is waste of time to check every frame in a long video to sort the important parts. To detect important parts in a frame, the autoencoder is used for this project. Using this model, we can extract the important parts in a frame and make a video time shorter which includes only import events in a video. We can apply this project in various fields. With an unsupervised approach, we have the advantage that there is no requirement for human annotations to learn the important event in a video. With this method, the evaluation shows that the process for video summarization has two summarized videos that are an important event and an unimportant event.

Paper: [Unsupervised video summarization using ITL-Autoencoder](https://gijunglee.github.io/assets/video_report.pdf)

## Steps
![Steps](/data/Picture1.png)
- The first step is to extract frames in a video
- The second step is generating pseudo labels for the frames
  - Inforamation_theoretic Learning-Autoencoder (ITL-AE)
- The third step is to classify actual frames by comparing with reconstruction scores and pseudo labels
- The fourth step is generating a summarized video
### Algorithms
![Algorithms](/data/Picture2.png)
<img src="/data/Picture2.png" width="700" height="500">
## Results
![result](/data/Picture3.png)

| Methods | Accuracy |
| -------- | -------- |
| PL | 65.33% |
| PL + RL | 87.44% |

Results of video summarization. PL: Pseudo labels, RL: Reconstruction loss

### Confusion Matrix
![result2](/data/Picture4.png)



## Dataset
- The brackish dataset contains 89 videos are provided with annotations in the AAU Bounding Box, YOLO Darknet, and MS COCO formats. Fish are annotated in six coarse categories. Categories: Big fish, Small fish, Crab, Shrimp, Jellyfish, Starfish.
- Paper: [Detection of Marine Animals in a New Underwater Dataset with Varying Visibility](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AAMVEM/Pedersen_Detection_of_Marine_Animals_in_a_New_Underwater_Dataset_with_CVPRW_2019_paper.pdf)
- Data: [The Brackish Dataset](https://www.kaggle.com/aalborguniversity/brackish-dataset)

![img](/data/Picture5.png)
