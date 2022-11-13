import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import cv2

#From confusion matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


#Parameters to set each fun
with_subevents = True
all_frames = True
DEBUG = 0 #note debug only runs one iteration
run_name=f"with_subevents_{with_subevents}_all_frames_{all_frames}" #all_frames_with_subevents"


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" #NOTE: DELETE THIS WHEN GPU AVAILABLE
model, preprocess = clip.load("ViT-B/32", device=device)
# template = "A picture of a person "

template = "A person is "
words = ["cooking","drinking","eating","exercising","getting up", "laying", "napping", "playing","reading","using something","watching TV", "writing"]
# sentances = {"Cook.Cut"             :"cooking by cutting something.",
#             "Cook.Usemicrowave"     :"cooking using a microwave",
#             "Cook.Useoven"          :"cooking using an oven",
#             "Cook.Usestove"         :"cooking using a stove",
#             "Drink.Frombottle"      :"drinking from a bottle",
#             "Drink.Fromcup"         :"drinking from a cup",
#             "Eat.Snack"             :"eating a snack",
#             "Eat.Useutensil"        :"eating using a utensil",
#             "Exercise"              :"exercising",
#             "Getup"                 :"getting up",
#             "Lay.Onbed"             :"laying on a bed",
#             "Nap"                   :"napping",
#             "Play.Boardgame"        :"playing a boardgame",
#             "Read"                  :"reading",
#             "Use.Coffeemachine"     :"using a coffee machine",
#             "Use.Computer"          :"using a computer",
#             "Use.Dishwasher"        :"using a dishwasher",
#             "Use.Gamecontroller"    :"using a gname controller",
#             "Use.Kettle"            :"using a kettle",
#             "Use.Mop"               :"using a mop",
#             "Use.Phone"             :"using a phone",
#             "Use.Refrig"            :"using a refrigerator",
#             "Use.Shelf"             :"using a shelf",
#             "Use.Sink"              :"using a sink",
#             "Use.Switch"            :"using a ninetendo switch",
#             "Use.Tablet"            :"using a tablet",
#             "Use.Vaccum"            :"using a vaccum",
#             "Watch.TV"              :"watching TV",
#             "Write"                 :"writing"
#             }
#need to be parallel arrays to maintain order (traversing dict does not maintain order...)
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
            "using a gname controller",
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
gt_labels = ["Cook.Cut"             ,
            "Cook.Usemicrowave"     ,
            "Cook.Useoven"          ,
            "Cook.Usestove"         ,
            "Drink.Frombottle"      ,
            "Drink.Fromcup"         ,
            "Eat.Snack"             ,
            "Eat.Useutensil"        ,
            "Exercise"              ,
            "Getup"                 ,
            "Lay.Onbed"             ,
            "Nap"                   ,
            "Play.Boardgame"        ,
            "Read"                  ,
            "Use.Coffeemachine"     ,
            "Use.Computer"          ,
            "Use.Dishwasher"        ,
            "Use.Gamecontroller"    ,
            "Use.Kettle"            ,
            "Use.Mop"               ,
            "Use.Phone"             ,
            "Use.Refrig"            ,
            "Use.Shelf"             ,
            "Use.Sink"              ,
            "Use.Switch"            ,
            "Use.Tablet"            ,
            "Use.Vaccum"            ,
            "Watch.TV"              ,
            "Write"                             
            ]
if with_subevents: words = sentances
text = clip.tokenize([template + w for w in words]).to(device)


class YouHomeDataset(torch.utils.data.Dataset):
    """
   using help from : https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
    """
    def __init__(self):
        #extract just gt labels
        base = "/home/abhi/research/SmartHome/Data/youhome_mp4_data/mp4data"
        labels_text = gt_labels
        if not with_subevents:
            new_list = []
            for f in labels_text:
                event=f.split(".")[0]
                if event not in new_list:
                    new_list.append(event)
            labels_text = new_list
        self.debug_labels_text = np.array(labels_text) #for debugging outside of this

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
ground_truths =[]
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
        if DEBUG: print("logits_img",logits_per_image.shape)
        if DEBUG: print("logits_text",logits_per_text.shape)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if DEBUG: print("probs.shape",probs.shape)
        pred = np.argmax(probs,axis=-1) #if we have a batch of n, we should have n maxes
        if DEBUG: print("pred shape", pred.shape, pred)
        if DEBUG: print("labels shape", labels.shape, labels)

        predictions = np.concatenate((predictions,pred))
        ground_truths = np.concatenate((ground_truths,labels))
        correct_count+=np.sum(pred==labels.cpu().numpy())
        # if correct_count >=0: print("corrent prediction!")
        # print("predictions shape",predictions.shape)

        if DEBUG: break
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
acc = correct_count/len(predictions)
print('Accuracy=',acc)
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

saved = np.vstack((ground_truths,predictions))
print("saved array size = ",saved.shape)

np.save("array_saves/"+run_name+'.npy',saved)


cm = metrics.confusion_matrix(ground_truths,predictions,labels=list(range(len(dataset.debug_labels_text))))
# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(30,30))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(dataset.debug_labels_text, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(dataset.debug_labels_text, fontsize = 10)
plt.yticks(rotation=0)

plt.title(run_name+f"_Acc:{acc}", fontsize=20)

plt.savefig(run_name+'.png')
plt.show()


