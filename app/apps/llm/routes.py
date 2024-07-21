import fastapi

from .schemas import PromptRequest
from apps.llm.services import TensorRTLLMEngine

router = fastapi.APIRouter(prefix="/llm", tags=["llm"])


llm_engine = TensorRTLLMEngine()


@router.post("/prompt")
async def prompt(request: PromptRequest):
    response = llm_engine.run_inference(request.prompt)
    return {"response": response}
