import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_hamster = preprocess(Image.open("CLIP.jpg")).unsqueeze(0).to(device)
image = preprocess(Image.open("../Data/testframes/testing_0.jpg")).unsqueeze(0).to(device)
image = torch.cat((image_hamster,image),dim=0)
print("new stacked image shape",image.shape)
print("shape:",image.shape)
# text = clip.tokenize([ "a dog walking", "a cat sleeping", "a hamster eating a carrot", "a pig eating a carrot", "a squirrel riding a bike"]).to(device)
text = clip.tokenize(["sleeping","walking","dancing","eating", "A picture of a person cooking."]).to(device)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


#note to self, try templating our youhome dataset.