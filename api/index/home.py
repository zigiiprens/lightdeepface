from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.encoders import jsonable_encoder

routerHome = APIRouter()


@routerHome.get("/")
async def get_root():
    """
    :param: none.
    :return: json response.
    """
    response = jsonable_encoder({"message": "<h1>Hello demoapi from LightDeepFace!</h1>"})
    return JSONResponse(status_code=status.HTTP_200_OK, content=response)
