import asyncio
import logging
from contextlib import asynccontextmanager
from multiprocessing import Manager, Process, Queue

import fastapi
from apps.llm.routes import router as llm_router
from apps.transcription.handlers import user_websocket_send_handler
from apps.transcription.routes import router as transcription_router
from core import exceptions
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config, db


def setup_processes(app: fastapi.FastAPI):
    clients = Manager().dict()
    vad_queues = {"source": Queue(), "result": Queue()}
    transcription_queues = {"source": Queue(), "result": Queue()}
    llm_queues = {"source": Queue(), "result": Queue()}
    config.Settings().clients = clients
    config.Settings().vad_queues = vad_queues
    config.Settings().transcription_queues = transcription_queues
    config.Settings().llm_queues = llm_queues

    from apps.transcription.services import run_llm, run_transcription, run_vad

    processes = [
        Process(
            target=run_vad, args=(clients, vad_queues["source"], vad_queues["result"])
        ),
        Process(
            target=run_transcription,
            args=(
                clients,
                transcription_queues["source"],
                transcription_queues["result"],
            ),
        ),
        Process(
            target=run_llm, args=(clients, llm_queues["source"], llm_queues["result"])
        ),
    ]
    app.state.processes = processes
    for process in processes:
        process.start()


async def setup_connection_handlers(app: fastapi.FastAPI):
    transcription_sent_queue = asyncio.Queue()
    summary_sent_queue = asyncio.Queue()

    app.state.handlers = [
        asyncio.create_task(
            user_websocket_send_handler(transcription_sent_queue, "transcription")
        ),
        asyncio.create_task(user_websocket_send_handler(summary_sent_queue, "summary")),
    ]

    config.Settings().transcription_sent_queue = transcription_sent_queue
    config.Settings().summary_sent_queue = summary_sent_queue


def empty_queue(queue: Queue):
    while not queue.empty():
        queue.get()


def terminate_processes(processes: list[Process]):
    empty_queue(config.Settings().vad_queues["source"])
    empty_queue(config.Settings().vad_queues["result"])
    empty_queue(config.Settings().transcription_queues["source"])
    empty_queue(config.Settings().transcription_queues["result"])
    empty_queue(config.Settings().llm_queues["source"])
    empty_queue(config.Settings().llm_queues["result"])

    sources = [
        config.Settings().vad_queues["source"],
        config.Settings().transcription_queues["source"],
        config.Settings().llm_queues["source"],
    ]
    for source_queue in sources:
        source_queue.put(("terminate", None, None))

    for process in processes:
        # process.terminate()
        process.join()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):  # type: ignore
    """Initialize application services."""
    await db.init_db()
    config.Settings.config_logger()

    setup_processes(app)
    await setup_connection_handlers(app)

    logging.info("Startup complete")
    yield

    terminate_processes(app.state.processes)
    for handler in app.state.handlers:
        handler.cancel()
    logging.info("Shutdown complete")


app = fastapi.FastAPI(
    title="Live Note",
    # description=DESCRIPTION,
    version="0.1.0",
    contact={
        "name": "Ali Molaee",
        "url": "https://github.com/Alima88/LiveNote",
        "email": "ali8molaee@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/Alima88/LiveNote/blob/main/LICENSE",
    },
    lifespan=lifespan,
)


@app.exception_handler(exceptions.BaseHTTPException)
async def base_http_exception_handler(
    request: fastapi.Request, exc: exceptions.BaseHTTPException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "error": exc.error},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: fastapi.Request, exc: Exception):
    import traceback

    traceback_str = "".join(traceback.format_tb(exc.__traceback__))
    # body = request._body

    logging.error(f"Exception: {traceback_str} {exc}")
    logging.error(f"Exception on request: {request.url}")
    # logging.error(f"Exception on request: {await request.body()}")
    return JSONResponse(
        status_code=500,
        content={"message": str(exc), "error": "Exception"},
    )


origins = [
    "http://localhost:8000",
    "https://wln.inbeet.tech",
    "https://app.wln.inbeet.tech",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm_router)
app.include_router(transcription_router)


@app.get("/")
async def index():
    return {"message": "Hello World!"}
