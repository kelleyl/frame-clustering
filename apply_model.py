import pickle
import joblib
from cluster import load_images, apply_pretrained, ModelBottom
from torchvision import models

if __name__ == '__main__':
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_model", action="store", dest="model_path",
                        help="path to kmeans model")
    parser.add_argument('--data', action="store", dest="data",
                        help='path to directory containing image dataset')
    parser.add_argument('--output', action="store", dest="out_csv",
                        help='path to directory containing image dataset')
    args = parser.parse_args()

    KMeans = joblib.load(args.model_path)

    # load pretrained model and chop off final layers
    vgg_model = models.vgg19(pretrained=True)
    model = ModelBottom(vgg_model)

    # preprocess data
    data = load_images(args.data)
    res, paths = apply_pretrained(model, data)
    class_dict = {}
    with open(args.out_csv, 'w') as out:
        out_writer = csv.writer(out)
        for embedding, filepath in zip(res, paths):
            _class = KMeans.predict(embedding.reshape(1,-1))[0]
            class_score = KMeans.transform(embedding.reshape(1,-1))[0]
            class_dict[_class] = class_dict.get(_class, 0) + 1
            row = [filepath, _class]
            row.extend(class_score)
            out_writer.writerow(row)
            out.flush()

    for c, count in class_dict.items():
        print (c, count)

