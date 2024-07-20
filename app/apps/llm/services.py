import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import tensorrt_llm
import torch
from tensorrt_llm.builder import get_engine_version
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer, PreTrainedTokenizer

from server.config import Settings


def read_model_name(engine_dir: str) -> tuple[str, Optional[str]]:
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", "r") as f:
        config = json.load(f)

    if engine_version is None:
        return config["builder_config"]["name"], None

    model_arch = config["pretrained_config"]["architecture"]
    model_version = None
    if model_arch == "ChatGLMForCausalLM":
        model_version = config["pretrained_config"]["chatglm_version"]
    return model_arch, model_version


def throttle_generator(generator: Generator, stream_interval: int) -> Generator:
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(
    tokenizer_dir: Optional[str] = None,
    vocab_file: Optional[str] = None,
    model_name: str = "GPTForCausalLM",
    model_version: Optional[str] = None,
    tokenizer_type: Optional[str] = None,
) -> tuple[PreTrainedTokenizer, int, int]:
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            tokenizer_type=tokenizer_type,
            use_fast=use_fast,
        )
    elif model_name == "GemmaForCausalLM":
        from transformers import GemmaTokenizer

        tokenizer = GemmaTokenizer(
            vocab_file=vocab_file,
            padding_side="left",
            truncation_side="left",
            legacy=False,
        )
    else:
        from transformers import T5Tokenizer

        tokenizer = T5Tokenizer(
            vocab_file=vocab_file,
            padding_side="left",
            truncation_side="left",
            legacy=False,
        )

    if model_name == "QWenForCausalLM":
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config["chat_format"]
        if chat_format == "raw" or chat_format == "chatml":
            pad_id = gen_config["pad_token_id"]
            end_id = gen_config["eos_token_id"]
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == "ChatGLMForCausalLM" and model_version == "glm":
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def clean_phi2_output(output: str) -> str:
    return output.split("Instruct:")[0]


def clean_phi3_output(output: str) -> str:
    return output.split("")[0]


class TensorRTLLMEngine:

    def __init__(
        self,
        model_path: str = Settings.phi_tensorrt_path,
        tokenizer_path: str = Settings.phi_tokenizer_path,
        **kwargs,
    ) -> None:
        self.log_level: str = kwargs.get("log_level", "error")
        self.runtime_rank: int = tensorrt_llm.mpi_rank()
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.pad_id: Optional[int] = None
        self.end_id: Optional[int] = None
        self.prompt_template: Optional[str] = None
        self.runner_cls = ModelRunner
        self.runner_kwargs: dict = {}
        self.runner: Optional[ModelRunner] = None
        self.last_prompt: Optional[str] = None
        self.last_output: Optional[list[str]] = None
        self.phi_model_type: Optional[str] = None
        self.chat_format: Optional[
            Union[
                Callable[[str, list[tuple[str, str]]], str],
                Callable[[str, list[tuple[str, str]], str], str],
            ]
        ] = None
        self.infer_time: Optional[float] = None
        self.eos: Optional[bool] = None

        self.initialize_model(model_path, tokenizer_path)

    def initialize_model(self, engine_dir: str, tokenizer_dir: str) -> None:
        logger.set_level(self.log_level)
        model_name, model_version = read_model_name(engine_dir)
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=None,
            model_name=model_name,
            model_version=model_version,
            tokenizer_type=None,
        )
        self.runner_kwargs = {
            "engine_dir": engine_dir,
            "lora_dir": None,
            "rank": self.runtime_rank,
            "debug_mode": False,
            "lora_ckpt_source": "hf",
        }
        self.runner = self.runner_cls.from_dir(**self.runner_kwargs)

    def parse_input(
        self,
        input_text: Optional[list[str]] = None,
        add_special_tokens: bool = True,
        max_input_length: int = 923,
        pad_id: Optional[int] = None,
    ) -> list[torch.Tensor]:
        if self.pad_id is None:
            self.pad_id = self.tokenizer.pad_token_id if self.tokenizer else None

        batch_input_ids: list[list[int]] = []
        for curr_text in input_text or []:
            if self.prompt_template is not None:
                curr_text = self.prompt_template.format(input_text=curr_text)
            input_ids = self.tokenizer.encode(
                curr_text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=max_input_length,
            )
            batch_input_ids.append(input_ids)

        batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
        return batch_input_ids

    def decode_tokens(
        self,
        output_ids: torch.Tensor,
        input_lengths: list[int],
        sequence_lengths: torch.Tensor,
        transcription_queue: Any,
    ) -> Optional[list[str]]:
        batch_size, num_beams, _ = output_ids.size()
        for batch_idx in range(batch_size):
            if transcription_queue.qsize() != 0:
                return None

            inputs = output_ids[batch_idx][0][: input_lengths[batch_idx]].tolist()
            input_text = self.tokenizer.decode(inputs, skip_special_tokens=False)
            output = []
            for beam in range(num_beams):
                if transcription_queue.qsize() != 0:
                    return None

                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                output_text = self.tokenizer.decode(outputs, skip_special_tokens=False)
                output.append(output_text)
        return output

    def format_prompt_chatml(
        self,
        prompt: str,
        conversation_history: list[tuple[str, str]],
        system_prompt: str = "",
    ) -> str:
        messages = []
        for user_prompt, llm_response in conversation_history:
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def format_prompt_qa(
        self, prompt: str, conversation_history: list[tuple[str, str]]
    ) -> str:
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Instruct: {user_prompt}\nOutput:{llm_response}\n"
        return f"{formatted_prompt}Instruct: {prompt}\nOutput:"

    def format_prompt_chat(
        self, prompt: str, conversation_history: list[tuple[str, str]]
    ) -> str:
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Alice: {user_prompt}\nBob:{llm_response}\n"
        return f"{formatted_prompt}Alice: {prompt}\nBob:"

    def run(
        self,
        model_path: str,
        tokenizer_path: str,
        phi_model_type: Optional[str] = None,
        transcription_queue: Any = None,
        llm_queue: Any = None,
        audio_queue: Any = None,
        input_text: Optional[list[str]] = None,
        max_output_len: int = 100,
        max_attention_window_size: int = 4096,
        num_beams: int = 1,
        streaming: bool = False,
        streaming_interval: int = 4,
        debug: bool = False,
    ) -> None:
        self.phi_model_type = phi_model_type
        if self.phi_model_type == "phi-2":
            self.chat_format = self.format_prompt_qa
        else:
            self.chat_format = self.format_prompt_chatml
        self.initialize_model(model_path, tokenizer_path)

        logging.info("[LLM INFO:] Loaded LLM TensorRT Engine.")

        conversation_history: dict = {}

        while True:
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue

            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output["prompt"].strip()

            if self.last_prompt == prompt:
                if self.last_output is not None and transcription_output["eos"]:
                    self.eos = transcription_output["eos"]
                    llm_queue.put(
                        {
                            "uid": transcription_output["uid"],
                            "llm_output": self.last_output,
                            "eos": self.eos,
                            "latency": self.infer_time,
                        }
                    )
                    audio_queue.put({"llm_output": self.last_output, "eos": self.eos})
                    conversation_history[transcription_output["uid"]].append(
                        (
                            transcription_output["prompt"].strip(),
                            self.last_output[0].strip(),
                        )
                    )
                    continue

            input_text = [
                self.chat_format(
                    prompt, conversation_history[transcription_output["uid"]]
                )
            ]

            self.eos = transcription_output["eos"]

            batch_input_ids = self.parse_input(
                input_text=input_text,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
            )

            input_lengths = [x.size(0) for x in batch_input_ids]

            logging.info(
                f"[LLM INFO:] Running LLM Inference with WhisperLive prompt: {prompt}, eos: {self.eos}"
            )
            start = time.time()
            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_output_len,
                    max_attention_window_size=max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    temperature=1.0,
                    top_k=1,
                    top_p=0.0,
                    num_beams=num_beams,
                    length_penalty=1.0,
                    repetition_penalty=1.0,
                    stop_words_list=None,
                    bad_words_list=None,
                    lora_uids=None,
                    prompt_table_path=None,
                    prompt_tasks=None,
                    streaming=streaming,
                    output_sequence_lengths=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()
            if streaming:
                for curr_outputs in throttle_generator(outputs, streaming_interval):
                    output_ids = curr_outputs["output_ids"]
                    sequence_lengths = curr_outputs["sequence_lengths"]
                    output = self.decode_tokens(
                        output_ids, input_lengths, sequence_lengths, transcription_queue
                    )

                    if output is None:
                        break

                if output is None:
                    continue
            else:
                output_ids = outputs["output_ids"]
                sequence_lengths = outputs["sequence_lengths"]
                if self.runner.gather_context_logits:
                    outputs["context_logits"]
                if self.runner.gather_generation_logits:
                    outputs["generation_logits"]
                output = self.decode_tokens(
                    output_ids, input_lengths, sequence_lengths, transcription_queue
                )
            self.infer_time = time.time() - start

            if output is not None:
                if self.phi_model_type == "phi-2":
                    output[0] = clean_phi2_output(output[0])
                self.last_output = output
                self.last_prompt = prompt
                llm_queue.put(
                    {
                        "uid": transcription_output["uid"],
                        "llm_output": output,
                        "eos": self.eos,
                        "latency": self.infer_time,
                    }
                )
                audio_queue.put({"llm_output": output, "eos": self.eos})
                logging.info(
                    f"[LLM INFO:] Output: {output[0]}\nLLM inference done in {self.infer_time} ms\n\n"
                )

            if self.eos:
                conversation_history[transcription_output["uid"]].append(
                    (transcription_output["prompt"].strip(), output[0].strip())
                )
                self.last_prompt = None
                self.last_output = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
