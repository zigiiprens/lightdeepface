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
from verify import vggface_model, facenet_model, openface_model, deepface_model, deepid_model, arcface_model

"""TF version settings"""
if tf_version == 1:
    config = tf.ConfigProto(device_count={'XLA_GPU': 0})
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = False
    graph = tf.get_default_graph()
    sess = tf.Session(config=config)

"""Router settings"""
routerAnalyze = APIRouter()


class VerifyInput(BaseModel):
    model_name: str
    distance_metric: str
    img: Optional[list]


routerVerify = APIRouter()


@routerVerify.post("/", summary="Verify Face")
async def verify_face(input_item: VerifyInput):
    global graph

    tic = time.time()
    req_model = input_item.model_name
    req_dist = input_item.distance_metric
    req_img = input_item.img
    trx_id = uuid.uuid4()

    # resp_obj = jsonable_encoder({'success': False})

    if tf_version == 1:
        with graph.as_default():
            resp_obj = verifyWrapper(req_model, req_dist, req_img, trx_id)
    elif tf_version == 2:
        resp_obj = verifyWrapper(req_model, req_dist, req_img, trx_id)

    # --------------------------

    toc = time.time()

    resp_obj["trx_id"] = trx_id
    resp_obj["seconds"] = toc - tic

    response = jsonable_encoder(resp_obj)
    return JSONResponse(status_code=status.HTTP_200_OK, content=response)


def verifyWrapper(req_model, req_distance, req_images, trx_id=None):
    resp_obj = JSONResponse({'success': False})

    model_name = "FaceNet"
    distance_metric = "cosine"
    if req_model:
        model_name = req_model
    if req_distance:
        distance_metric = req_distance

    # ----------------------

    instances = []
    if req_images:
        raw_content = req_images  # list

        for item in raw_content:  # item is in type of dict
            instance = []
            img1 = item["img1"]
            img2 = item["img2"]

            validate_img1 = False
            if len(img1) > 11 and img1[0:11] == "data:image/":
                validate_img1 = True

            validate_img2 = False
            if len(img2) > 11 and img2[0:11] == "data:image/":
                validate_img2 = True

            if validate_img1 != True or validate_img2 != True:
                return JSONResponse(status_code=205, content=jsonable_encoder(
                    {'success': False, 'error': 'you must pass both img1 and img2 as base64 encoded string'}))

            instance.append(img1)
            instance.append(img2)
            instances.append(instance)

    # --------------------------

    if len(instances) == 0:
        return JSONResponse(status_code=205, content=jsonable_encoder({'success': False, 'error': 'you must pass at least one img object in your request'}))

    print("Input request of ", trx_id, " has ", len(instances), " pairs to verify")

    # --------------------------

    if model_name == "VGG-Face":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=vggface_model)
    elif model_name == "FaceNet":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=facenet_model)
    elif model_name == "OpenFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=openface_model)
    elif model_name == "DeepFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=deepface_model)
    elif model_name == "DeepID":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=deepid_model)
    elif model_name == "ArcFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=arcface_model)
    elif model_name == "Ensemble":
        models = {}
        # models["VGG-Face"] = vggface_model
        models["Facenet"] = facenet_model
        # models["OpenFace"] = openface_model
        # models["DeepFace"] = deepface_model
        models["ArcFace"] = arcface_model
        resp_obj = DeepFace.verify(instances, model_name=model_name, model=models)
    else:
        resp_obj = jsonable_encoder({'success': False, 'error': 'You must pass a valid model name. You passed %s' % (model_name)})

    return resp_obj
