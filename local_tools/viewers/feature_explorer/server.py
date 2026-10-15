import argparse
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

import interpretation_api
from heatmap_api import router as heatmap_router
from whole_api import router as whole_router


ROOT = Path(__file__).parent
torch.set_num_threads(2)

app = FastAPI(docs_url=None, redoc_url=None)
app.include_router(whole_router)
app.include_router(heatmap_router)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
app.mount('/shared', StaticFiles(directory=ROOT.parent / 'shared'), name='shared')


@app.middleware('http')
async def private_response(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin and origin != 'http://' + request.headers['host']:
        return JSONResponse({'detail': 'Local origin required'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@app.get('/')
def index():
    return RedirectResponse('/whole')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'service': 'sclvmi-heatmaps', 'campaign': interpretation_api.SETTINGS['campaign']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
