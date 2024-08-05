import asyncio
import logging
import uuid

import fastapi

from .handlers import connected_clients
from .services import VoiceReceiverService

router = fastapi.APIRouter(tags=["transcription"])


@router.websocket("/ws/{client_id}")
async def get_audio(websocket: fastapi.WebSocket, client_id: uuid.UUID):
    client_id = str(client_id)
    await websocket.accept()
    connected_clients[client_id] = websocket
    logging.debug(f"WebSocket connection established for {client_id}")
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                await VoiceReceiverService().no_voice_activity(client_id, -1)
                continue

            if message["type"] == "websocket.connect":
                raise ValueError(
                    f"Unexpected message type: websocket.connect {client_id}"
                )
            elif message["type"] == "websocket.receive":
                if "bytes" in message:
                    data = message["bytes"]
                    await VoiceReceiverService().process_audio_frame(data, client_id)
                elif "text" in message:
                    data = message["text"]
                    if data == "close()":
                        await VoiceReceiverService().no_voice_activity(client_id, -1)
                        await websocket.close()
                        break
                else:
                    raise ValueError(f"Unexpected message format {message.keys()}")

            elif message["type"] == "websocket.disconnect":
                await VoiceReceiverService().no_voice_activity(client_id, -1)
                # await websocket.close()
                break
            else:
                raise ValueError(
                    f"Unexpected message type: {message['type']} from {client_id}"
                )
    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))

        logging.error(f"WebSocket connection closed for {client_id}: {e}")
        logging.error(f"Exception: {traceback_str} {e}")
    finally:
        VoiceReceiverService().pop_clients_audios(client_id)
        connected_clients.pop(client_id, None)
