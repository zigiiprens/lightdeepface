import argparse
import uuid
import json
import time
from tqdm import tqdm

from deepface import DeepFace

"""Load Facial Attribute Analysis Models"""
tic = time.time()

print("Loading Facial Attribute Analysis Models...")

pbar = tqdm(range(0, 4), desc='Loading Facial Attribute Analysis Models...')

for index in pbar:
    if index == 0:
        pbar.set_description("Loading emotion analysis model")
        emotion_model = DeepFace.build_model('Emotion')
    elif index == 1:
        pbar.set_description("Loading age prediction model")
        age_model = DeepFace.build_model('Age')
    elif index == 2:
        pbar.set_description("Loading gender prediction model")
        gender_model = DeepFace.build_model('Gender')
    elif index == 3:
        pbar.set_description("Loading race prediction model")
        race_model = DeepFace.build_model('Race')

toc = time.time()

facial_attribute_models = {}
facial_attribute_models["emotion"] = emotion_model
facial_attribute_models["age"] = age_model
facial_attribute_models["gender"] = gender_model
facial_attribute_models["race"] = race_model

print("Facial attribute analysis models are built in ", toc - tic, " seconds")
