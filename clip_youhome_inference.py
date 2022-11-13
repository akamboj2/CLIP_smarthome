import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from scipy import stats


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# template = "A picture of a person "
with_subevents = True
all_frames = True
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

#extract just gt labels
base = "/home/abhi/research/SmartHome/Data/youhome_mp4_data/mp4data"
one_p = os.listdir(os.path.join(base,'p101'))
labels = []
for opt in one_p:
    if with_subevents:
        gt = opt
    else:
        gt = opt.split('.')[0]
    if gt not in labels:
        labels.append(gt)

# currently length is 12
assert len(labels)==len(words), f"number of ground truth labels {len(labels)} need to equal number of predicted labels{len(words)}"

#now extract test images from files
test_imgs = []
test_gt = []
for p in os.listdir(base):
    for opt in os.listdir(os.path.join(base,p)):
        vids = os.listdir(os.path.join(base,p,opt))
        for v in vids:
            if v.split("_")[-1]=='frames':
                frames = os.listdir(os.path.join(base,p,opt,v))
                if all_frames:
                    test_imgs.append([os.path.join(base,p,opt,v,take_frame) for take_frame in frames])
                else:
                    #currently let's try taking just the middle frame
                    take_frame = len(frames)//2
                    test_imgs.append([os.path.join(base,p,opt,v,frames[take_frame])])
                
                if with_subevents: 
                    test_gt.append(labels.index(opt))
                else: 
                    test_gt.append(labels.index(opt.split('.')[0]))

print("number test images",len(test_imgs))

#now using image files, perform inference with clip and print the number correct
correct_count = 0
predictions=[]
for ind, imgs in tqdm(enumerate(test_imgs)):
    per_frame_predictions = []
    for img_file in imgs:
        # print("on image:",img_file)
        image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = np.argmax(probs)
        per_frame_predictions.append(pred)
    
    pred = stats.mode(per_frame_predictions)
    predictions.append(pred)
    if pred==test_gt[ind]:
        correct_count+=1
    # print("predicted",labels[pred], "gt", labels[test_gt[ind]])
    # break

print('number correct:',correct_count)
print('total number', len(predictions))
print('Accuracy=',correct_count/len(predictions))
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

np.save("predictions_single_mid_all_sent.npy",np.array(predictions))

#note to self, try templating our youhome dataset.