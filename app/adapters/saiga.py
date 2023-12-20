from threading import Thread
from typing import Iterator

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    TextIteratorStreamer
)

from app.adapters.base import LLMAdapterBase


MESSAGE_TEMPLATE = '<s>{role}\n{content}</s>\n'
SYSTEM_PROMPT = 'Ты — АТОМ, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.'
START_TOKEN_ID = 1
BOT_TOKEN_ID = 9225
MODEL_NAME = 'IlyaGusev/saiga2_7b_lora'


class Conversation:
    """Formats the prompt from the chat messages."""

    def __init__(
        self,
        message_template: int = MESSAGE_TEMPLATE,
        system_prompt: int = SYSTEM_PROMPT,
        start_token_id: int = START_TOKEN_ID,
        bot_token_id: int = BOT_TOKEN_ID,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            'role': 'system',
            'content': system_prompt
        }]

    def get_start_token_id(self) -> int:
        return self.start_token_id

    def add_user_message(self, message: str) -> None:
        self.messages.append({
            'role': 'user',
            'content': message
        })

    def add_bot_message(self, message: str) -> None:
        self.messages.append({
            'role': 'bot',
            'content': message
        })

    def get_prompt(self, tokenizer: AutoTokenizer) -> str:
        final_text = ''
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


class SaigaAdapter(LLMAdapterBase):

    def __init__(self, llm: PreTrainedModel, model_id: str = MODEL_NAME) -> None:
        super().__init__(llm, model_id, adapter_name='saiga')

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.generation_config = GenerationConfig.from_pretrained(model_id)

    def preformat(self, message: str, history: list[tuple[str, str]]) -> str:
        chat = Conversation()
        for human, assistant in history:
            chat.add_user_message(human)
            chat.add_bot_message(assistant)
        chat.add_user_message(message)
        return chat.get_prompt(self.tokenizer)

    def generator(self, llm: PreTrainedModel, prompt: str) -> Iterator[str]:
        data = self.tokenizer(prompt, return_tensors='pt')
        data = {k: v.to(llm.device) for k, v in data.items()}
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = dict(
            **data,
            generation_config=self.generation_config,
            streamer=streamer,
        )
        stream = Thread(target=llm.generate, kwargs=generation_kwargs)
        stream.start()
        return streamer
