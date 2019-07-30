import json
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#import matplotlib.pyplot as plt
import numpy as np
import torch


from commons import get_model, get_tensor

with open('cat_to_name.json') as f:
	cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
	class_to_idx = json.load(f)


idx_to_class = {v:k for k, v in class_to_idx.items()}

model = get_model()

# def get_flower_name(image_bytes):
    # tensor = get_tensor(image_bytes)
    # outputs = model.forward(tensor)
    # _, prediction = outputs.max(1)
    # category = prediction.item()
    # class_idx = idx_to_class[category]
    # flower_name = cat_to_name[class_idx]
    # return category, flower_name



def predict(tensor, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    flower_names = []
    outputs = model.forward(tensor)
    probs, prediction = outputs.topk(5)

    top_probs = []
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    for i in range(0,5):
        cat = prediction[0,i].item()
        odds = torch.exp(probs[0,i])
        prob = odds/(1+odds)
        top_probs.append(prob.item())
        class_idx = idx_to_class[cat]
        flower_names.append(cat_to_name[class_idx])
    top_flower_name = flower_names[0]
    return top_flower_name, flower_names, top_probs
    

    
# def plot_solution(tensor):

    # fig = plt.figure(figsize = (6,10))
    # plt.subplot(2,1,1)
    # npimg = tensor.numpy()[0,:,:,:]
    # plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    # plt.subplot(2,1,2)
    # probs, flowers = predict(tensor) 
    # y_pos = np.arange(len(flowers))
    # plt.barh(y_pos, probs)
    # plt.yticks(y_pos, flowers)
    # plt.savefig("static/temp/output.png")

    # return flowers,probs