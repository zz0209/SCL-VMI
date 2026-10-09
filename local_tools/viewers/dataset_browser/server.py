import argparse
import ipaddress
import json
from pathlib import Path

import plotly
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from store import DatasetStore


workspace = Path(__file__).resolve().parents[3]
network = json.loads((workspace / "local_tools/network.local.json").read_text(encoding="utf-8"))
allowed_networks = [ipaddress.ip_network(value) for value in network["allowed_networks"]]
store = DatasetStore(workspace)
app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver", network["lan_address"]])
static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")
app.mount("/shared", StaticFiles(directory=Path(__file__).parent.parent / "shared"), name="shared")


@app.middleware("http")
async def local_responses(request: Request, call_next):
    if request.client.host != "testclient" and not any(ipaddress.ip_address(request.client.host) in subnet for subnet in allowed_networks):
        return JSONResponse({"detail": "LAN access required"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin != "http://"+request.headers["host"]:
        return JSONResponse({"detail": "Local origin required"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(KeyError)
async def missing_resource(request, error):
    return JSONResponse({"detail": str(error)}, status_code=404)


@app.exception_handler(ValueError)
async def invalid_data(request, error):
    return JSONResponse({"detail": str(error)}, status_code=422)


@app.get("/")
def index():
    return FileResponse(static / "index.html")


@app.get("/plotly.js")
def plotly_script():
    return FileResponse(Path(plotly.__file__).parent / "package_data/plotly.min.js", media_type="application/javascript")


@app.get("/api/catalog")
def catalog():
    return store.catalog()


@app.get("/api/cases/{dataset}")
def cases(dataset: str, search: str = "", role: str = "all", label: str = "all", offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=100)):
    return store.list_cases(dataset, search, role, label, offset, limit)


@app.get("/api/case/{dataset}/{case}")
def describe(dataset: str, case: str, channel: int = 0):
    return store.describe(dataset, case, channel)


@app.get("/api/slice/{dataset}/{case}")
def slice_image(dataset: str, case: str, channel: int = 0, axis: int = 2, index: int = 0, low: float = -1000, high: float = 400, overlay: bool = True, region: int = -1):
    return Response(store.slice_png(dataset, case, channel, axis, index, low, high, overlay, region), media_type="image/png")


@app.get("/api/mesh/{dataset}/{case}")
def mesh(dataset: str, case: str, channel: int = 0, region: int = -1):
    return store.mesh(dataset, case, channel, region)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=network["listen_host"], port=args.port, access_log=False, proxy_headers=False)
