import asyncio
import logging
from datetime import datetime

import fastapi
from json_advanced import dumps

connected_clients: dict[str, fastapi.WebSocket] = {}


async def user_websocket_send_handler(queue: asyncio.Queue, queue_name: str):
    logging.debug(f"Starting handler for {queue_name}")

    while True:
        new_task: tuple[str, str, datetime] = await queue.get()
        client_id, text, at = new_task

        ws: fastapi.WebSocket = connected_clients.get(client_id)
        if ws:
            msg = dumps({"type": queue_name, "data": text})
            await ws.send_text(msg)
