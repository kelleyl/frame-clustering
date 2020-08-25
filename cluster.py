import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import math
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image


class ModelBottom(nn.Module):
    '''
    class to create a new model by removing final layers of an existing model
    '''
    def __init__(self, original_model):
        super(ModelBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1]) ## this needs to be modified for different models
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    ## https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def load_images(root):
    normalize = transforms.Compose([            #[1]
         transforms.Resize(299),                    #[2]
         transforms.CenterCrop(299),                #[3]
         transforms.ToTensor(),                     #[4]
         transforms.Normalize(                      #[5]
         mean=[0.485, 0.456, 0.406],                #[6]
         std=[0.229, 0.224, 0.225]                  #[7]
     )])
    data_set = ImageFolderWithPaths(root=root, transform=normalize)
    data = DataLoader(data_set, batch_size=1, num_workers=2, shuffle=True)
    return data

def apply_pretrained(model, data, count=math.inf):
    '''
    apply a model to a given number of instances, applies to the first 'count' instances passed in data
    :param model:
    :param data:
    :param count:
    :return:
    '''
    res = []
    paths = []
    for i, d in tqdm(enumerate(data)):
        data = d[0]
        path = d[2]
        if (i > count):
            break
        out = model.forward(data)
        flat = out.flatten()
        res.append(flat.numpy())
        paths.append(path)
    return res, paths

if __name__ == '__main__':
    import os
    import pickle
    parser = argparse.ArgumentParser(description='Model Training ')
    parser.add_argument('--data', action="store", dest="data",
                        help='path to directory containing image dataset')
    parser.add_argument('--model', action="store", dest="model",
                        help="pretrained model name")
    parser.add_argument('--output', action="store", dest="output",
                        help="location to store the output pickle")
    parser.add_argument('--data-count', action="store", dest="inst_count",
                        default=math.inf, help="number of instances to use for cluster training")

    parser.add_argument('--data-store', action="store", dest="data_embed",
                        help='If this param is used, data and model are ignored,'
                             'data-store should contain a tuple where the first element is a'
                             'numpy array where each row is an embedding and the second element'
                             'is a list of filenames.')

    parser.add_argument('--cluster_model', action="store", dest="cluster_store",
                        default=None, help="location to store the KMeans model.")

    parser.add_argument('--vis', action="store", dest="vis", default=True,
                        help="generate output for tensorboard visualization")

    args = parser.parse_args()
    res, paths = None, None

    if args.data_embed and os.path.isfile(args.data_embed):
        res, paths = pickle.load(open(args.data_embed, 'rb'))
    elif args.data and os.path.isdir(args.data):
        data = load_images(args.data)
        if args.model == "resnet50":
            res50_model = models.resnet50(pretrained=True)
            model = ModelBottom(res50_model)
        elif args.model == "vgg19":
            vgg_model = models.vgg19(pretrained=True)
            model = ModelBottom(vgg_model)
        else:
            raise BaseException("Invalid Model Argument")

        res, paths = apply_pretrained(model, data, count=args.inst_count)
        pickle.dump((res, paths), open(args.output, 'wb'))

    if args.vis:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
        writer.add_embedding(np.array(res), metadata=paths)
        writer.close()

    if args.cluster_store:
        from sklearn.cluster import KMeans
        import joblib
        print ("clustering...")
        clustering = KMeans(n_clusters=75, random_state=11).fit(res)

        print ("saving model...")
        joblib.dump(clustering, args.cluster_store)

        unique, counts = np.unique(clustering.labels_, return_counts=True)
        print (dict(zip(unique, counts)))



