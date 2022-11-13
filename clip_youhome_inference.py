import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" #NOTE: DELETE THIS WHEN GPU AVAILABLE
model, preprocess = clip.load("ViT-B/32", device=device)
# template = "A picture of a person "
with_subevents = False
all_frames = False
template = "A person is "
words = ["cooking","drinking","eating","exercising","getting up", "laying", "napping", "playing","reading","using something","watching TV", "writing"]
sentances = ["cooking by cutting something.",
            "cooking using a microwave",
            "cooking using an oven",
            "cooking using a stove",
            "drinking from a bottle",
            "drinking from a cup",
            "eating a snack",
            "eating using a utensil",
            "exercising",
            "getting up",
            "laying on a bed",
            "napping",
            "playing a boardgame",
            "reading",
            "using a coffee machine",
            "using a computer",
            "using a dishwasher",
            "using a game controller",
            "using a kettle",
            "using a mop",
            "using a phone",
            "using a refrigerator",
            "using a shelf",
            "using a sink",
            "using a ninetendo switch",
            "using a tablet",
            "using a vaccum",
            "watching TV",
            "writing"
            ]
if with_subevents: words = sentances
text = clip.tokenize([template + w for w in words]).to(device)


class YouHomeDataset(torch.utils.data.Dataset):
    """
   using help from : https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
    """
    def __init__(self):
        #extract just gt labels
        #NOTE: I'm assuming p101 contains all event labels in its subfolder
        base = "/home/abhi/research/SmartHome/Data/youhome_mp4_data/mp4data"
        one_p = os.listdir(os.path.join(base,'p101'))
        labels_text = []
        for opt in one_p:
            if with_subevents:
                gt = opt
            else:
                gt = opt.split('.')[0]
            if gt not in labels_text:
                labels_text.append(gt)
        self.debug_labels_text = np.array(labels_text)

        # currently length is 12
        assert len(labels_text)==len(words), f"number of ground truth labels {len(labels_text)} need to equal number of predicted labels{len(words)}"

        #now extract test images from files
        self.test_imgs = []
        self.test_gt = []
        for p in os.listdir(base):
            for opt in os.listdir(os.path.join(base,p)):
                vids = os.listdir(os.path.join(base,p,opt))
                for v in vids:
                    if v.split("_")[-1]=='frames':
                        frames = os.listdir(os.path.join(base,p,opt,v))
                        if all_frames:
                            #add all the frames of the video (with action labels) to dataset
                            for take_frame in frames:
                                self.test_imgs.append(os.path.join(base,p,opt,v,take_frame))
                                if with_subevents: 
                                    self.test_gt.append(labels_text.index(opt))
                                else: 
                                    self.test_gt.append(labels_text.index(opt.split('.')[0]))
                        else:
                            #currently let's try taking just the middle frame from the video
                            take_frame = len(frames)//2
                            self.test_imgs.append(os.path.join(base,p,opt,v,frames[take_frame]))
                            if with_subevents: 
                                self.test_gt.append(labels_text.index(opt))
                            else: 
                                self.test_gt.append(labels_text.index(opt.split('.')[0]))

        print("number test images",len(self.test_gt))

    def __len__(self):
        return len(self.test_gt)

    def __getitem__(self,idx):
        img_file = self.test_imgs[idx] #NOTE: I wonder if this slows it down, bc we processing images as we get 
                                        # items as opposed to reading all the images in the beginning init
        image = preprocess(Image.open(img_file)).to(device)
        return (image,img_file),self.test_gt[idx] #passing img_file for debugging



#now using image files, perform inference with clip and print the number correct
correct_count = 0
predictions=[]
dataset = YouHomeDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# for imgs, labels in dataloader:
#     print("Batch of images has shape: ",imgs.shape)
#     print("Batch of labels has shape: ", labels.shape)
for ind, ((images,files),labels) in tqdm(enumerate(dataloader)):
    # print(type(images)) #it's a torch tensor
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print("probs.shape",probs.shape)
        pred = np.argmax(probs,axis=-1) #if we have a batch of n, we should have n maxes
        # print("pred shape", pred.shape, pred)
        # print("labels shape", labels.shape, labels)

        predictions = np.concatenate((predictions,pred))
        correct_count+=np.sum(pred==labels.cpu().numpy())
        # if correct_count >=0: print("corrent prediction!")
        # print("predictions shape",predictions.shape)

        # break
        #for debugging:
        """OKAY this makes it seem like whatever preprcoess() CLIP code is doing is making the image really hard to see and distinguish
        objects ? Maybe try without this preprocessing? """
        # print("debugging:")
        # for i in range(images.shape[0]):
        #     #this should loop 32 times, one for every elt in batch
        #     print("file: ",files[i])
        #     print("predicted",dataset.debug_labels_text[pred[i]], "gt", dataset.debug_labels_text[labels[i]])
        #     cv2.imshow('img',images[i].permute((1,2,0)).cpu().numpy()) # permute so it's (h,w,c)
        #     cv2.imshow('img RGB to BRG transformed',images[i].permute((1,2,0)).cpu().numpy()[:,:,::-1])
        #     cv2.imshow('img without preprocessing',np.array(Image.open(files[i])))
        #     while(True):
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     print()
        # break

print('number correct:',correct_count)
print('total number', len(predictions))
print('Accuracy=',correct_count/len(predictions))
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# np.save("predictions_single_mid_all_sent.npy",np.array(predictions))











"""
NOTE:
DEBUGGING: I don't think the dataset is working now, because if i run it, it should be giving a 9% accuracy (as i tested before),
BUt none of the images are being classified correctly! 
FIRST reread this code and make sure it makes sense and i didn't make a silly typo error
-> it probably is this because even random guessing should have gotten better results.

So to debug i propose, pull the code from the repo i pushed
run it
find which img is being classified correctly (also maybe visualize the image as well to see if it's the preprocess that's derping)
and then break this loop when we read that exact image file, and see what is being classified as.
If you need to, save the np array outputed from the dataset, and the one outputted from the correct version.
it literally should be the same bits going through the same image model so not sure why none of the images here are being classified correctly


best regards, 
abhi kamboj

"""