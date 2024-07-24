import logging
import uuid

import fastapi

from .services import VoiceReceiverService
from .vad import VoiceActivityDetection

router = fastapi.APIRouter(prefix="/transcription", tags=["transcription"])

connected_clients = {}

vad_model = VoiceActivityDetection()


@router.websocket("/ws/{client_id}")
async def get_audio(websocket: fastapi.WebSocket, client_id: uuid.UUID):
    client_id = str(client_id)
    await websocket.accept()
    connected_clients[client_id] = websocket
    logging.info(f"WebSocket connection established for {client_id}")
    try:
        while True:
            data = await websocket.receive_bytes()
            if data == b"close()":
                VoiceReceiverService().no_voice_activity(client_id, -1)
                await websocket.close()
                break

            await VoiceReceiverService().process_audio_frame(data, client_id)

    except Exception as e:
        logging.error(f"WebSocket connection closed for {client_id}: {e}")
    finally:
        # Remove the client from the dictionary when disconnected
        del connected_clients[client_id]
