import time
from tqdm import tqdm

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])

from deepface import DeepFace


tic = time.time()

print("Loading Face Recognition Models...")

pbar = tqdm(range(0, 6), desc='Loading Face Recognition Models...')

for index in pbar:

    if index == 0:
        pbar.set_description("Loading VGG-Face")
        vggface_model = DeepFace.build_model("VGG-Face")
    elif index == 1:
        pbar.set_description("Loading OpenFace")
        openface_model = DeepFace.build_model("OpenFace")
    elif index == 2:
        pbar.set_description("Loading Google FaceNet")
        facenet_model = DeepFace.build_model("Facenet")
    elif index == 3:
        pbar.set_description("Loading Facebook DeepFace")
        deepface_model = DeepFace.build_model("DeepFace")
    elif index == 4:
        pbar.set_description("Loading DeepID DeepFace")
        deepid_model = DeepFace.build_model("DeepID")
    elif index == 5:
        pbar.set_description("Loading ArcFace DeepFace")
        arcface_model = DeepFace.build_model("ArcFace")

toc = time.time()

print("Face recognition models are built in ", toc - tic, " seconds")
