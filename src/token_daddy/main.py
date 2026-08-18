from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI

from token_daddy.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
    yield
    await app.state.redis.aclose()


app = FastAPI(title="token_daddy", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/healthz/redis")
async def healthz_redis():
    pong = await app.state.redis.ping()
    return {"redis": "ok" if pong else "unreachable"}


def main():
    import uvicorn

    uvicorn.run("token_daddy.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
