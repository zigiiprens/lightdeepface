from typing import Optional

from fastapi import APIRouter, status, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import uuid
import time
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
from deepface import DeepFace
from analyze import *


"""TF version settings"""
if tf_version == 1:
    config = tf.ConfigProto(device_count={'XLA_GPU': 0})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = False
    graph = tf.get_default_graph()
    # sess_emotion = tf.Session(config=config)

"""Router settings"""
routerAnalyze = APIRouter()


class AnalyzeInput(BaseModel):
    img: Optional[list]


@routerAnalyze.post("/", summary="Analyze Face")
async def analyze_face(input_item: AnalyzeInput):
    global graph

    m_tic: float = time.time()
    req_img = input_item.img
    trx_id = uuid.uuid4()

    if tf_version == 1:
        with graph.as_default():
            resp_obj = analyzeWrapper(req_img, trx_id)
    elif tf_version == 2:
        resp_obj = analyzeWrapper(req_img, trx_id)

    m_toc: float = time.time()

    resp_obj["trx_id"] = trx_id
    resp_obj["seconds"] = m_toc - m_tic
    response = jsonable_encoder({"message": "Analyze", "response": resp_obj})
    return JSONResponse(status_code=status.HTTP_200_OK, content=response)


def analyzeWrapper(req, trx_id=0):
    resp_obj = JSONResponse({'success': False})

    instances = []
    # if "img" in list(req.keys()):
    #     raw_content = req["img"]  # list

    for item in req:  # item is in type of dict
        instances.append(item)

    if len(instances) == 0:
        return JSONResponse(status_code=205, content=jsonable_encoder({'success': False, 'error': 'you must pass at least one img object in your request'}))

    print("Analyzing ", len(instances), " instances")

    # ---------------------------

    actions = ['emotion', 'age', 'gender', 'race']
    # if "actions" in list(req.keys()):
    #     actions = req["actions"]

    # ---------------------------

    # with tf.Session(config=config):
    # resp_obj = DeepFace.analyze(instances, actions=actions)
    resp_obj = DeepFace.analyze(instances, actions=actions, models=facial_attribute_models)

    return resp_obj
