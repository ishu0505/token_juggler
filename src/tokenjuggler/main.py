"""Optional thin HTTP service. Mark 1 is library-first: services import
tokenjuggler and share quotas through one Redis. This app only reports health
and usage for that Redis; an HTTP generate endpoint is future work."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from tokenjuggler.client import TokenJuggler


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    app.state.tj = TokenJuggler.from_config(os.environ.get("TOKENJUGGLER_CONFIG", "tokenjuggler.yaml"))
    yield
    await app.state.tj.aclose()


app = FastAPI(title="tokenjuggler", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/usage")
async def usage(hours: float = 24):
    from datetime import timedelta

    return await app.state.tj.usage(all_projects=True, since=timedelta(hours=hours))


def serve():
    import uvicorn

    uvicorn.run("tokenjuggler.main:app", host="0.0.0.0", port=8000)
