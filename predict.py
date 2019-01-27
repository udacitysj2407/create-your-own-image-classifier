
# coding: utf-8

# In[ ]:


import argparse
import helper as hp
import json
import numpy as np
parser = argparse.ArgumentParser(description = 'predict.py')
parser.add_argument('--gpu_cpu', dest= "gpu_cpu", action = "store", default = "gpu", help = "enable gpu computation")
parser.add_argument('filepath', default='./flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, help='top k predictions')
parser.add_argument('--category_names', default='./cat_to_name.json', help='mapping the folder number to the flower category')
pa = parser.parse_args()
filepath = pa.filepath
gpu_cpu = pa.gpu_cpu
checkpoint=pa.checkpoint
top_k=pa.top_k
category_names=pa.category_names

train_data,trainloader, validationloader, testloader= hp.load_data()
model = hp.load_checkpoint_rebuild_model(checkpoint)
probs = hp.predict(filepath, model, top_k, gpu_cpu)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
top_prob = np.array(probs[0][0])
top_categories =  [cat_to_name[str(i + 1)] for i in np.array(probs[1][0])]
i=0
while i < top_k:
    print("{} with a probability of {}".format(top_categories[i], top_prob[i]))
    i += 1

