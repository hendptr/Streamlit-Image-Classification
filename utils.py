import torch 
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image 


cuda = torch.cuda.is_available()



def predict(Network, img, num, device):
    #image = Image.open(img).convert('RGB')
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
                                    
    image = transform(img).unsqueeze(0).to(device)

    Network.to(device)
    Network.eval()


    out = Network(image)
    probability, label = torch.topk(out, num)
    if cuda:
        label = label.cpu().data.numpy()
        probability = F.softmax(probability).cpu().data.numpy() * 100
    else:
        label = label.data.numpy()
        probability = F.softmax(probability).data.numpy() * 100

    with open("labels.txt") as f:
        idx2label = eval(f.read())

    data = [(probability[0][i], label[0][i]) for i in range(num)]
    data = [(idx2label[data[i][1]], data[i][0]) for i in range(num)]
    return data
