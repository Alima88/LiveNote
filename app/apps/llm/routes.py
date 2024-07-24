import fastapi

from .schemas import PromptRequest

router = fastapi.APIRouter(prefix="/llm", tags=["llm"])


# llm_engine = TensorRTLLMEngine()


@router.post("/prompt")
async def prompt(request: PromptRequest):
    # response = llm_engine.run_inference(request.prompt)
    response = "test"
    return {"response": response}
