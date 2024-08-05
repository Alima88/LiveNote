import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue

import fastapi
from json_advanced import dumps

connected_clients: dict[str, fastapi.WebSocket] = {}


async def user_websocket_send_handler(queue: asyncio.Queue, queue_name: str):
    while True:
        new_task: tuple[str, str, datetime, dict] = await queue.get()
        client_id, text, _, _ = new_task

        logging.debug(f"Sending {queue_name} to {client_id} {text}")

        ws: fastapi.WebSocket = connected_clients.get(client_id)
        if ws:
            msg = dumps({"type": queue_name, "data": text})
            await ws.send_text(msg)


async def handle_transcription_result(
    transcription_queue_result: Queue,
    transcription_sent_queue: asyncio.Queue,
    llm_queue: Queue,
) -> None:
    logging.info(f"started")
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        while True:
            client_id, text, res_metadata = await loop.run_in_executor(
                executor, transcription_queue_result.get
            )
            await transcription_sent_queue.put(
                (client_id, text, datetime.now(), res_metadata)
            )

            if llm_queue.qsize() == 1:
                new_task: tuple[str, str, datetime, dict] = await loop.run_in_executor(
                    executor, llm_queue.get
                )
                client_id, text, at, metadata = new_task
                if metadata.get("eos"):
                    llm_queue.put((client_id, text, at, metadata))

            if res_metadata.get("eos"):
                llm_queue.put((client_id, text, datetime.now(), res_metadata))
            elif llm_queue.qsize() == 0 and text.count(" ") > 10:
                logging.debug(f"llm_queue is empty {client_id}, {text}")
                llm_queue.put((client_id, text, datetime.now(), res_metadata))

            # asyncio.create_task(handle_llm_result(client_id))


async def handle_llm_result(
    llm_queue_result: Queue, summary_sent_queue: asyncio.Queue
) -> None:
    logging.info(f"started")
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        while True:
            client_id, summary, at, metadata = await loop.run_in_executor(
                executor, llm_queue_result.get
            )
            await summary_sent_queue.put((client_id, summary, at, metadata))
