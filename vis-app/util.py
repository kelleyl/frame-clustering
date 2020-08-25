import csv
import os
import pickle
import time
from operator import itemgetter
from collections import defaultdict
from sklearn.metrics import pairwise
from glob import glob

def load_clusters(filename):
    with open(filename) as fn:
        rows = csv.DictReader(fn, fieldnames=["filename", "cluster_id"]+["c_{}".format(i) for i in range(75)])
        # print (list(rows)[0])
        return {r["filename"]:(r["cluster_id"], r["c_{}".format(r["cluster_id"])]) for r in rows}

def load_scenes_csv(PATH="static/img/shots/"):
    files_dict = {}
    for f in glob(PATH + '*.csv'):
        video_fname = f[len(PATH):-11]
        video_fname = video_fname[:9] + '_' + video_fname[10:]
        shot_dict = {}

        with open(f) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            next(reader)

            for row in reader:
                shot_dict[row[0]] = row[3]
            files_dict[video_fname] = shot_dict
    return files_dict

def load_cosine_similarity_matrix(filename):
    '''
    :param filename: path to pickle containing path list and embedding matrix
    :return: matrix of cosine similarities, list of paths
    if the cosine similarity matrix has already been computed for a given embedding matrix
    then load the existing cosine similarity matrix, otherwise generate then save the cosine
    similarity matrix.
    '''
    cosine_filename = filename+".cosine.pkl"
    res, paths = pickle.load(open(filename, 'rb'))
    if os.path.exists(cosine_filename):
        print ("loading cosine sim matrix...")
        cs_matrix = pickle.load(open(cosine_filename, 'rb'))
    else:
        print ("generating cosine sim matrix...")
        cs_matrix = pairwise.cosine_similarity(res)
        pickle.dump(cs_matrix, open(cosine_filename, 'wb'))
    return cs_matrix, paths

print ("loading files...")
start = time.time()
# file_cluster_dict = load_clusters("static/indices/all_output.csv")
# cosine_sim_matrix, paths = load_cosine_similarity_matrix("static/indices/all_vgg19.pkl")
cosine_sim_matrix, paths = load_cosine_similarity_matrix("static/indices/newshour_shot_inception.pkl")

# id_filename = {v:k for v, k in enumerate(list(map(lambda x: x[0][x[0].rindex(os.path.sep)+1:], paths)))}
id_filename = {v:k for v, k in enumerate(paths)}

filename_id = {filename: id for id, filename in id_filename.items()}
print ("done loading")
print (f"time:{ time.time() - start }")
#
# def query(cluster_id):
#     ## get only dicts that are for this cluster_id
#     c_list = [(filename[filename.rindex(os.path.sep)+1:-3], dist[1]) for (filename, dist) in file_cluster_dict.items() if dist[0] == cluster_id]
#     return c_list

def query_doc(doc_filename):
    doc_index = filename_id[doc_filename]
    index_score = sorted(enumerate(cosine_sim_matrix[doc_index]), key=itemgetter(1), reverse=True)
    return [id_filename[index] for index, _ in index_score], [score for _, score in index_score]
