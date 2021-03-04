from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from index.home import routerHome
from analyze.analyze import routerAnalyze
from verify.verify import routerVerify

import uvicorn

# Tags metadata
tags_metadata = [
    {
        "name": "index",
        "description": "Operations with index index.",
    },
    {
        "name": "analyze",
        "description": "Operations with analyze.",
    },
    {
        "name": "verify",
        "description": "Operations with verify.",
    }
]

origins = [
    "http://127.0.0.1:8888",
]

# FastAPI initialization app
app = FastAPI(
    title="demoapi",
    description="This project is a demo project, created to the simple use to serve models.",
    openapi_tags=tags_metadata,
    version="0.0.1"
)

# App static folder {for css files}
# app.mount("/static", StaticFiles(directory="static"), name="static")


# App add middleware CORS, allows origins, credentials, methods and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins
)

# App include routers {users, items, admin and index}
app.include_router(
    routerVerify,
    prefix="/verify",
    tags=["verifier"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    routerAnalyze,
    prefix="/analyze",
    tags=["analyzer"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    routerHome,
    tags=["index"],
    responses={404: {"description": "Not found"}},
)

if __name__ == "__main__":
    uvicorn.run("api2:app", host="127.0.0.1", port=8888)
