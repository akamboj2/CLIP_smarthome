# Commands to Run:
# python clip_youhome_inference.py --data_dir /home/abhi/research/SmartHome/Data/full_data --num_classes 31 | tee log_outs/2.txt
#new dataset --data_dir /home/abhi/research/SmartHome/Data/imgdata4event_2021_full_cropped
# generalization dataset --data_dir /home/abhi/research/SmartHome/Data/person_generalize




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
from torchvision import transforms

from tensorboardX import SummaryWriter
from pathlib import Path
import random
import argparse


#Parameters to set 
DEBUG = 0 #note debug only runs one iteration
parser = argparse.ArgumentParser(description='Youhome Clip Adaptation module')
parser.add_argument('--data_dir', '-d', type=str, default='/home/abhi/research/SmartHome/Data/full_data',
                    help='path to the dataset directory')
parser.add_argument('--num_epochs', '-e', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
parser.add_argument('--train',type=bool, default=True, help="Set to True when training, false when testing")
parser.add_argument('--with_subevents',type=bool, default=True, help="Set to True when detecting subevents as well")
parser.add_argument('--combine_text',type=bool, default=False, help="Set to True when combining Text with images as input from CLIP's output to adaptation module")
parser.add_argument('--use_adaptation_module',type=bool, default=True, help="Set to True when when using adaptation module")
parser.add_argument('--adapt_module',type=str, default='MLP', help="type of adaptation module to use - defaults to resnet")

args = parser.parse_args()

#set run name
if DEBUG:
    run_name='debugging'
else:
    run_name=f"test_with_subevents_{args.with_subevents}_lr_{args.learning_rate}_epochs_{args.num_epochs}_person_generalize" 
print("Running:", run_name)


# Setup labels as captions
template = "A person is "
words = ["cooking","drinking","eating","entering","exercising","getting up", "laying", "leaving","napping", "playing","reading","using something","watching TV", "writing"]
if args.num_classes==31:
    gt_labels = ["Cook.Cut"             ,
            "Cook.Usemicrowave"     ,
            "Cook.Useoven"          ,
            "Cook.Usestove"         ,
            "Drink.Frombottle"      ,
            "Drink.Fromcup"         ,
            "Eat.Snack"             ,
            "Eat.Useutensil"        ,
            "Enter"                 ,
            "Exercise"              ,
            "Getup"                 ,
            "Lay.Onbed"             ,
            "Leave"                 ,
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
    sentences = ["cooking by cutting something.",
            "cooking using a microwave",
            "cooking using an oven",
            "cooking using a stove",
            "drinking from a bottle",
            "drinking from a cup",
            "eating a snack",
            "eating using a utensil",
            "entering the room",
            "exercising",
            "getting up",
            "laying on a bed",
            "leaving the room",
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
else:
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
    sentences = ["cooking by cutting something",
        "cooking using a microwave",
        "cooking using an oven",
        "cooking using a stove",
        "drinking from a bottle",
        "drinking from a cup",
        "eating a snack",
        "eating using a utensil",
        "entering the room",
        "exercising",
        "getting up",
        "laying on a bed",
        "leaving the room",
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
assert args.num_classes== (len(gt_labels) if args.with_subevents else len(words)), \
    f"expected num_classes = number of ground truth labels: {args.num_classes}  = {gt_labels}"

#Log loss train and validation loss on Tensorboard
log_dir = Path('log_dir') / run_name 
log_dir.mkdir(parents=True, exist_ok=True)
train_writer = SummaryWriter(log_dir/'train')
val_writer = SummaryWriter(log_dir / 'val')

#Set up pretrained CLIP model and input labels
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
if args.with_subevents: words = sentences
text = clip.tokenize([template + w for w in words]).to(device)

#function to load dataset
def load_dataset():
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
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
        train_dataset_small, batch_size=args.batch_size, shuffle=True,
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
        batch_size=args.batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=False
    )

    return trainloader, valloader

dataloader_train, dataloader_test = load_dataset()
# maybe later
class EventModel(nn.Module):
    def __init__(self):
        super(EventModel, self).__init__()
        # self.foundation = CLIP()
        if args.adapt_module=='resnet':
            #resnet version
            self.adapt_model = torchvision.models.resnet18(weights=None) #randomly initialized weights
            num_features = adapt_model.fc.in_features #need to change output to match our number of classes
            self.adapt_model.fc = nn.Linear(num_features, len(gt_labels))
            self.adapt_model.conv1 = nn.Sequential(nn.Linear(512,32*32*3),nn.Unflatten(1,(3,32,32)),adapt_model.conv1)
        elif args.adapt_module=='MLP':
            # # MLP Version:
            if args.combine_text:
                self.adapt_model = torchvision.ops.MLP(args.num_classes,[256,128,args.num_classes]) #because clip's model.encode(0 results in batch_sizex512)
            else:
                self.adapt_model = torchvision.ops.MLP(512,[256,128,args.num_classes]) #because clip's model.encode(0 results in batch_sizex512)
        else:
            raise Exception(f"incorrect args.adapt_module {args.adapt_module}")

    def forward(self, x):
        return self.adapt_model(x)


#training adaption module
#https://medium.com/nerd-for-tech/image-classification-using-transfer-learning-pytorch-resnet18-32b642148cbe

# adapt_model = adapt_model.to(device)
adapt_model = EventModel().to(device)
criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(adapt_model.parameters(),lr=args.learning_rate, weight_decay=args.decay)#optim.SGD(adapt_model.parameters(),lr=learning_rate, momentum=.9) # lr=.001
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25)

#train loop
start_time = time.time() #(for showing time)
if args.train and args.use_adaptation_module:
    for epoch in range(args.num_epochs):
        adapt_model.train()
        running_loss = 0.
        running_corrects = 0
        for ind, (images,labels) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(text.to(device))
            labels = labels.to(device)
            if args.combine_text:
                image_features_normed = image_features/image_features.norm(dim=-1, keepdim=True)
                text_features_normed = text_features/text_features.norm(dim=-1, keepdim=True)
                image_features = image_features_normed @ text_features_normed.T
                

            if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)
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

            if ind %100==0:
                # name, value, iteration
                train_writer.add_scalar("loss",loss.item(),ind+epoch*len((dataloader_train)))

        epoch_loss = running_loss / len(dataloader_train.dataset)
        epoch_acc = running_corrects / len(dataloader_train.dataset) * 100
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() -start_time))
        train_writer.add_scalar("epoch_loss",epoch_loss,epoch)
        train_writer.add_scalar("epoch_acc",epoch_acc,epoch)

        # update the learning rate
        scheduler.step()

        print("Intermediate model save as ", 'models/'+run_name+'.pth')
        torch.save(adapt_model.state_dict(), 'models/'+run_name+'.pth')
        if epoch%5 ==0:
            #This loop is for performing testing/evaluation
            print("VALIDATING MODEL")
            adapt_model.eval()
            with torch.no_grad():
                running_loss = 0.
                running_corrects = 0
                for ind, (images,labels) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                    # print(type(images)) #it's a torch tensor
                    image_features = model.encode_image(images.to(device))
                    text_features = model.encode_text(text)
                    labels = labels.to(device)
                    if args.combine_text:
                        image_features_normed = image_features/image_features.norm(dim=-1, keepdim=True)
                        text_features_normed = text_features/text_features.norm(dim=-1, keepdim=True)
                        image_features = image_features_normed @ text_features_normed.T

                    if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)

                    outputs = adapt_model(image_features.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(dataloader_test.dataset)
                epoch_acc = running_corrects / len(dataloader_test.dataset) * 100.
                print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))
                val_writer.add_scalar("epoch_loss",epoch_loss,epoch)
                val_writer.add_scalar("epoch_acc",epoch_acc,epoch)
                if DEBUG: break


        
    print("Finished training, saving model as ", 'models/'+run_name+'.pth')
    torch.save(adapt_model.state_dict(), 'models/'+run_name+'.pth')


if not args.train:
    adapt_model.load_state_dict(torch.load('models/'+run_name+'.pth'))
    print("Successfully loaded:",'models/'+run_name+'.pth')
print("Performing final evaluation")

#Run last evaluation for the confusion matrix
adapt_model.eval()
with torch.no_grad():
    running_loss = 0.
    running_corrects = 0
    running_labels = np.array([])
    running_preds = np.array([])
    for ind, (images,labels) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
        # print(type(images)) #it's a torch tensor
        image_features = model.encode_image(images.to(device))
        text_features = model.encode_text(text)
        labels = labels.to(device)
        if args.combine_text or not args.use_adaptation_module:
            image_features_normed = image_features/image_features.norm(dim=-1, keepdim=True)
            text_features_normed = text_features/text_features.norm(dim=-1, keepdim=True)
            if args.combine_text:
                image_features = image_features_normed @ text_features_normed.T

        if DEBUG: print("image_features",image_features.shape,"text_features",text_features.shape)

        outputs = adapt_model(image_features.float()) if args.use_adaptation_module else (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, preds = torch.max(outputs, 1)
        running_preds = np.concatenate((running_preds,preds.cpu().numpy()))
        running_labels = np.concatenate((running_labels,labels.cpu().numpy()))
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader_test.dataset)
    epoch_acc = running_corrects / len(dataloader_test.dataset) * 100.
    # print('FINAL [Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))

#Confusion Matrix
if DEBUG: print("running_preds",running_preds.shape, "running_labels",running_labels.shape)
#https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python
cm = metrics.confusion_matrix(running_labels,running_preds,labels=list(range(args.num_classes)))
# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(30,30))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(np.array(gt_labels if args.with_subevents else words) , fontsize = 10)
ax.xaxis.tick_bottom()
ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(np.array(gt_labels if args.with_subevents else words), fontsize = 10)
plt.yticks(rotation=0)
plt.title(run_name+f"_Acc:{epoch_acc}", fontsize=20)
plt.savefig('CMs/'+run_name+'.png')
plt.show()




# TODO: 
# setup argparser
# attemp to make this a custom model and push that
# try another super lightweight convolution model, maybe even make your own
# if time is an issue, try using XACC
#   but according to here: https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/#How_do_GPUs_work
#   seems like 3080 always performs better than titan rtx despite titan having more gpu memory (24 vs 10 GB)


#IF it doesn't work:
# - try with a different train test split (move one p folder from test to train)
# - try USING the text embeddings, instead of predicting only from image. 
#       to do this, output batch_sizex512 vector than do the top5 style similarity score
    # OR multiply img embedding by text embedding, flatten that matrix and pass that through a few linear layers (MLP) 
    # and train that to output correct classes
# instead of sampling Every frame, sample every 10 frames or 25 frames (25 frames at 25 fps would be 1 hz)
# logistic regression with augmented data

""""
To Ask Neo:
- 31 events? i'm counting 29 in all the folders
- what is binary_react dataset?
- was it tested on Toyota dataset?

Ask Junhao?
- in load dataset what is the random subset for? why not use all the data?
- what are the bumps every 25 epochs in your train graph? - learning rate scheduler, every 25 epochs turn down learning rate.
- train-val split? - 98%
- - cross camera challenge, 3 cameras in one room recording the same activity at the same time but not doing the 
- - overfitting in our case may not be a problem, because people in the same home will look around the same and act about the same as well
- - paper: can you infer this action based on where the user is sitting -> context dependent
- - take two shots on the same activity and take the first as training and validation, understand generilizability of the model


what to aim for according to Junhao, train on other data as well. 
- imagine a use case, how do you apply this foundation model to. If we have thousands of households.  if we can use 
our data to create 3-5 house holds in 30 activities , in 5 homes, each home would do 7-8 activites
use one model to serve all 5 homes - make this scalable

generalizability - specific part of data set to a 
"""