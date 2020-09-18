import torch
from torchvision import transforms
from PIL import Image
import json
import sys
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pt', map_location=device)
model.eval()

data_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

path = sys.argv[1]

result = {}

for file_name in tqdm(os.listdir(path)):
    image = Image.open(os.path.join(path, file_name))
    
    s = 'male' if model(data_transforms(image).unsqueeze(0).to(device)).argmax().item() else 'female'
    result[file_name] = s
    
    with open('process_results.json', 'w') as file:
        json.dump(result, file)
