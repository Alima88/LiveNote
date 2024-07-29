import asyncio
import logging
import uuid

import fastapi

from .handlers import connected_clients
from .services import VoiceReceiverService

router = fastapi.APIRouter(prefix="/transcription", tags=["transcription"])


@router.websocket("/ws/{client_id}")
async def get_audio(websocket: fastapi.WebSocket, client_id: uuid.UUID):
    client_id = str(client_id)
    await websocket.accept()
    connected_clients[client_id] = websocket
    logging.debug(f"WebSocket connection established for {client_id}")
    try:
        while True:
            try:
                # data = await websocket.receive_bytes()
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                if data == b"close()":
                    await VoiceReceiverService().no_voice_activity(client_id, -1)
                    await websocket.close()
                    break

                await VoiceReceiverService().process_audio_frame(data, client_id)
            except asyncio.TimeoutError:
                await VoiceReceiverService().no_voice_activity(client_id, -1)
                logging.debug(
                    f"No audio frame received within timeout period for client {client_id}"
                )
                # Continue loop to retry reading from websocket

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))

        logging.error(f"WebSocket connection closed for {client_id}: {e}")
        logging.error(f"Exception: {traceback_str} {e}")
    finally:
        connected_clients.pop(client_id, None)
