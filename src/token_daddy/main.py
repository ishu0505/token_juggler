"""Optional thin HTTP service. Mark 1 is library-first: services import
token_daddy and share quotas through one Redis. This app only reports health
and usage for that Redis; an HTTP generate endpoint is future work."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from token_daddy.client import TokenDaddy


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    app.state.td = TokenDaddy.from_config(os.environ.get("TOKEN_DADDY_CONFIG", "token_daddy.yaml"))
    yield
    await app.state.td.aclose()


app = FastAPI(title="token_daddy", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/usage")
async def usage(hours: float = 24):
    from datetime import timedelta

    return await app.state.td.usage(all_projects=True, since=timedelta(hours=hours))


def serve():
    import uvicorn

    uvicorn.run("token_daddy.main:app", host="0.0.0.0", port=8000)
