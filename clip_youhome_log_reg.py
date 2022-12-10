#for baseline clip inference and analysis
import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import cv2

#for confusion matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

#for Adaptation module training
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import random

from sklearn.linear_model import LogisticRegression
# import clip_youhome_inference.load_dataset,clip_youhome_inference.args


#Parameters to set each fun
with_subevents = True
all_frames = True
top5 = False
batch_size = 256
DEBUG = 0 #note debug only runs one iteration
if DEBUG:
    run_name='debugging'
else:
    run_name=f"with_subevents_{with_subevents}_all_frames_{all_frames},top5{top5}" #all_frames_with_subevents"


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" #NOTE: DELETE THIS WHEN GPU AVAILABLE
model, preprocess = clip.load("ViT-B/32", device=device)
# template = "A picture of a person "

template = "A person is "
words = ["cooking","drinking","eating","exercising","getting up", "laying", "napping", "playing","reading","using something","watching TV", "writing"]

#needs to be parallel arrays (not dict) to maintain order (traversing dict does not maintain order...)
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


def load_dataset():
    traindir = os.path.join('/home/abhi/research/SmartHome/Data/image_folder_yolo_cropped_data', 'train')
    valdir = os.path.join('/home/abhi/research/SmartHome/Data/image_folder_yolo_cropped_data', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),

        # util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        # normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform=train_transforms
    )
    train_dataset_small = torch.utils.data.Subset(
        train_dataset, 
        random.sample(range(len(train_dataset)), k=int(len(train_dataset)/20)))

    trainloader = torch.utils.data.DataLoader(
        train_dataset_small, batch_size=batch_size, shuffle=True,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=False
    )

    val_dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            # normalize
        ]))
    val_dataset_small = torch.utils.data.Subset(
        val_dataset,
        random.sample(range(len(val_dataset)), k=int(len(val_dataset)/10)))

    valloader = torch.utils.data.DataLoader(
        val_dataset_small,
        batch_size=batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=False
    )

    return trainloader, valloader



#now using image files, perform inference with clip and print the number correct
correct_count = 0
predictions=[]
ground_truths =[]
train_loader, test_loader = load_dataset()
dataset_train = train_loader.dataset #YouHomeDataset("/home/abhi/research/SmartHome/Data/youhome_mp4_data/train_split")
dataset_test = test_loader.dataset #YouHomeDataset("/home/abhi/research/SmartHome/Data/youhome_mp4_data/test_split")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(torch.utils.data.DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(dataset_train)
test_features, test_labels = get_features(dataset_test)

#sweep cs to find best performance
vals = []
cs = np.arange(0,1.01,.01) #[.19]#
for c in cs:
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0,C=c, max_iter=1000, verbose=0) #C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"\nC={c} Accuracy = {accuracy:.3f}")
    vals.append(accuracy)
    
m = np.max(vals)
ind = np.argmax(vals)
print(f"Maximum accuracy of {m} at C={cs[ind]}")

# #This loop is for performing testing/evaluation
# for ind, ((images,files),labels) in tqdm(enumerate(dataloader)):
#     # print(type(images)) #it's a torch tensor
#     with torch.no_grad():
#         image_features = model.encode_image(images)
#         text_features = model.encode_text(text)
        
#         if top5:
#             # Pick the top 5 most similar labels for the image
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#             if DEBUG: print("img_features",image_features.shape,"txt_features",text_features.shape, "Similarity",similarity.shape)
#             values, indices = similarity.topk(5,dim=1) #returns [batch_size,5] tensors
#             if DEBUG: print("values",values.shape, "indices",indices.shape)
#             pred = indices[:,0].cpu().numpy()

#             if DEBUG:
#                 indices = indices[0]
#                 values = values[0,:]
#                 print("\nTop predictions:\n")
#                 for value, index in zip(values, indices):
#                     print(f"{gt_labels[index]:>16s}: {100 * value.item():.2f}%")
#                 print("Actual value", gt_labels[labels[0]])
#         else:
#             logits_per_image, logits_per_text = model(images, text)
#             if DEBUG: print("logits_img",logits_per_image.shape)
#             if DEBUG: print("logits_text",logits_per_text.shape)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#             if DEBUG: print("probs.shape",probs.shape)
#             pred = np.argmax(probs,axis=-1) #if we have a batch of n, we should have n maxes
#             if DEBUG: print("pred shape", pred.shape, pred)
#             if DEBUG: print("labels shape", labels.shape, labels)

#         predictions = np.concatenate((predictions,pred))
#         ground_truths = np.concatenate((ground_truths,labels))
#         correct_count+=np.sum(pred==labels.cpu().numpy())
#         # if correct_count >=0: print("corrent prediction!")
#         # print("predictions shape",predictions.shape)

#         if DEBUG: break
#         #for debugging:
#         """OKAY this makes it seem like whatever preprcoess() CLIP code is doing is making the image really hard to see and distinguish
#         objects ? Maybe try without this preprocessing? """
#         # print("debugging:")
#         # for i in range(images.shape[0]):
#         #     #this should loop 32 times, one for every elt in batch
#         #     print("file: ",files[i])
#         #     print("predicted",dataset.debug_labels_text[pred[i]], "gt", dataset.debug_labels_text[labels[i]])
#         #     cv2.imshow('img',images[i].permute((1,2,0)).cpu().numpy()) # permute so it's (h,w,c)
#         #     cv2.imshow('img RGB to BRG transformed',images[i].permute((1,2,0)).cpu().numpy()[:,:,::-1])
#         #     cv2.imshow('img without preprocessing',np.array(Image.open(files[i])))
#         #     while(True):
#         #         if cv2.waitKey(1) & 0xFF == ord('q'):
#         #             break
#         #     print()
#         # break




# print('number correct:',correct_count)
# print('total number', len(predictions))
# acc = correct_count/len(predictions)
# print('Accuracy=',acc)
# # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# saved = np.vstack((ground_truths,predictions))
# print("saved array size = ",saved.shape)
# np.save("array_saves/"+run_name+'.npy',saved)

# #https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python
# cm = metrics.confusion_matrix(ground_truths,predictions,labels=list(range(len(dataset.debug_labels_text))))
# # Plot confusion matrix in a beautiful manner
# fig = plt.figure(figsize=(30,30))
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# # labels, title and ticks
# ax.set_xlabel('Predicted', fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=90)
# ax.xaxis.set_ticklabels(dataset.debug_labels_text, fontsize = 10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('True', fontsize=20)
# ax.yaxis.set_ticklabels(dataset.debug_labels_text, fontsize = 10)
# plt.yticks(rotation=0)
# plt.title(run_name+f"_Acc:{acc}", fontsize=20)
# plt.savefig(run_name+'.png')
# plt.show()


