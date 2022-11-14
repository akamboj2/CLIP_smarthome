#for baseline clip inference and analysis
import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import cv2
import time

#for confusion matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

#for Adaptation module training
import torch.nn as nn
import torch.optim as optim
import torchvision



#Parameters to set each fun
with_subevents = True
all_frames = True
top5 = False #deprecated Should delete probable
batch_size = 64
num_epochs = 60

DEBUG = 0 #note debug only runs one iteration
if DEBUG:
    run_name='debugging'
else:
    run_name=f"resnet18_with_subevents_{with_subevents}_all_frames_{all_frames}" #all_frames_with_subevents"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
# template = "A picture of a person " #this shows minimal diference in results

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


class YouHomeDataset(torch.utils.data.Dataset):
    """
   using help from : https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
    """
    def __init__(self,base_dir):
        #extract just gt labels
        base = base_dir
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

        print("number of images",len(self.test_gt))

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
# dataset = YouHomeDataset("/home/abhi/research/SmartHome/Data/youhome_mp4_data/mp4data")
dataset_train = YouHomeDataset("/home/abhi/research/SmartHome/Data/youhome_mp4_data/train_split")
dataset_test = YouHomeDataset("/home/abhi/research/SmartHome/Data/youhome_mp4_data/test_split")
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# maybe later
# class EventModel(nn.Module):
#     def __init__(self):
#         super(EventModel, self).__init__()
#         self.base = torchvision.models.resnet18(pretrained=False)
#         self.in_head = 
#         self.out_head = 



#training adaption module
#https://medium.com/nerd-for-tech/image-classification-using-transfer-learning-pytorch-resnet18-32b642148cbe
adapt_model = torchvision.models.resnet18(pretrained=False)
num_features = adapt_model.fc.in_features #need to change output to match our number of classes
adapt_model.fc = nn.Linear(num_features, len(gt_labels))
adapt_model.conv1 = nn.Sequential(nn.Linear(512,32*32*3),nn.Unflatten(1,(3,32,32)),adapt_model.conv1)
adapt_model = adapt_model.to(device)
criterion =  nn.CrossEntropyLoss()
optimizer = optim.SGD(adapt_model.parameters(),lr=.001, momentum=.9)

start_time = time.time() #(for showing time)
for epoch in range(num_epochs):
    adapt_model.train()
    running_loss = 0.
    running_corrects = 0
    for ind, ((images,files),labels) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
        image_features = model.encode_image(images.to(device))
        text_features = model.encode_text(text.to(device))
        labels = labels.to(device)
        # if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)
        # image_features torch.Size([32, 512]) text_features torch.Size([29, 512])

        optimizer.zero_grad()
        outputs = adapt_model(image_features.float())
        _, preds = torch.max(outputs,1)
        loss = criterion(outputs,labels)

        #get loss and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item()* images.size(0) #is this a thing? weight by the batch size?
        running_corrects += torch.sum(preds == labels.data) #why labels.data here?

    epoch_loss = running_loss / len(dataset_train)
    epoch_acc = running_corrects / len(dataset_train) * 100
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() -start_time))


    if epoch%5 ==0:
        #This loop is for performing testing/evaluation
        adapt_model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for ind, ((images,files),labels) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                # print(type(images)) #it's a torch tensor
                image_features = model.encode_image(images)
                text_features = model.encode_text(text)
                labels = labels.to(device)
                if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)

                outputs = adapt_model(image_features.float())
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataset_test)
            epoch_acc = running_corrects / len(dataset_train) * 100.
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))

                    # if top5:
                    #     # Pick the top 5 most similar labels for the image
                    #     image_features /= image_features.norm(dim=-1, keepdim=True)
                    #     text_features /= text_features.norm(dim=-1, keepdim=True)
                    #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    #     if DEBUG: print("img_features",image_features.shape,"txt_features",text_features.shape, "Similarity",similarity.shape)
                    #     values, indices = similarity.topk(5,dim=1) #returns [batch_size,5] tensors
                    #     if DEBUG: print("values",values.shape, "indices",indices.shape)
                    #     pred = indices[:,0].cpu().numpy()

                    #     if DEBUG:
                    #         indices = indices[0]
                    #         values = values[0,:]
                    #         print("\nTop predictions:\n")
                    #         for value, index in zip(values, indices):
                    #             print(f"{gt_labels[index]:>16s}: {100 * value.item():.2f}%")
                    #         print("Actual value", gt_labels[labels[0]])
                    # else:
                    #     logits_per_image, logits_per_text = model(images, text)
                    #     if DEBUG: print("logits_img",logits_per_image.shape)
                    #     if DEBUG: print("logits_text",logits_per_text.shape)
                    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    #     if DEBUG: print("probs.shape",probs.shape)
                    #     pred = np.argmax(probs,axis=-1) #if we have a batch of n, we should have n maxes
                    #     if DEBUG: print("pred shape", pred.shape, pred)
                    #     if DEBUG: print("labels shape", labels.shape, labels)

                    # predictions = np.concatenate((predictions,pred))
                    # ground_truths = np.concatenate((ground_truths,labels))
                    # correct_count+=np.sum(pred==labels.cpu().numpy())
                    # # if correct_count >=0: print("corrent prediction!")
                    # # print("predictions shape",predictions.shape)

            if DEBUG: break


    
print("Finished training, saving model as ", 'models/'+run_name+'.pth')
torch.save(model.state_dict(), 'models/'+run_name+'.pth')

#NOTE: No need to do all of this already given in eval code above! and don't save all the predictions! Already saving model, just rerun if u need predictions
# print('number correct:',correct_count)
# print('total number', len(predictions))
# acc = correct_count/len(predictions)
# print('Accuracy=',acc)
# # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# saved = np.vstack((ground_truths,predictions))
# print("saved array size = ",saved.shape)
# np.save("array_saves/"+run_name+'.npy',saved)


print("Performing evaluation")
#Run one last evaluation for the confusion matrix
adapt_model.eval()
with torch.no_grad():
    running_loss = 0.
    running_corrects = 0
    running_labels = np.array([])
    running_preds = np.array([])
    for ind, ((images,files),labels) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
        # print(type(images)) #it's a torch tensor
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        labels = labels.to(device)
        if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)

        outputs = adapt_model(image_features.float())
        _, preds = torch.max(outputs, 1)
        running_preds = np.concatenate((running_preds,preds.cpu().numpy()))
        running_labels = np.concatenate((running_labels,labels.cpu().numpy()))
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataset_test)
    epoch_acc = running_corrects / len(dataset_train) * 100.
    print('FINAL [Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))


#https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python
cm = metrics.confusion_matrix(running_labels,running_preds,labels=list(range(len(gt_labels))))
# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(30,30))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(gt_labels, fontsize = 10)
ax.xaxis.tick_bottom()
ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(gt_labels, fontsize = 10)
plt.yticks(rotation=0)
plt.title(run_name+f"_Acc:{epoch_acc}", fontsize=20)
plt.savefig('CMs/'+run_name+'.png')
plt.show()




# TODO: 
# First Clean up all the comments in this section, everything is pushed so nothing is lost!
# log on tensorboard
# setup argparser
# attemp to make this a custom model and push that
# try another super lightweight convolution model, maybe even make your own


#IF it doesn't work:
# - try with a different train test split (move one p folder from test to train)
# - try USING the text embeddings, instead of predicting only from image. 
#       to do this, output batch_sizex512 vector than do the top5 style similarity score