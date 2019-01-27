
# coding: utf-8

# In[ ]:


import argparse
import helper as hp

parser = argparse.ArgumentParser(description = 'predict.py')
parser.add_argument('--gpu_cpu', dest= "gpu_cpu", action = "store", default = "gpu", help = "enable gpu computation")
parser.add_argument('filepath', default='flowers/test/5/image_05159.jpg', help='Path to image to predict on.')
parser.add_argument('checkpoint_filepath', metavar='checkpoint_filepath', help='Path to model checkpoint to predict with.')
parser.add_argument('--top_k', default=5, help='top k predictions')
parser.add_argument('--category_names', default='./cat_to_name.json', help='mapping the folder number to the flower category')


args=parser.parse_args()
model = hp.load_checkpoint_rebuild_model(args.checkpoint_filepath)
probs, classes = hp.predict(args.filepath, model, args.gpu_cpu, args.top_k)

top_categories = [cat_to_name[str(i)] for i in classes]




