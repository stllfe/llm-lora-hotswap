from threading import Thread
from typing import Iterable, Iterator

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    TextIteratorStreamer,
    StoppingCriteria
)

from app.adapters.base import LLMAdapterBase


MESSAGE_TEMPLATE = '### {role}: {content}'
STOP_PHRASE = '### Human:'
SYSTEM_PROMPT = 'Your name is ATOM, you are an english-speaking automatic assistant. You speak to people and assist them.'
MODEL_NAME = 'kaitchup/Llama-2-7B-oasstguanaco-adapter'


class Conversation:
    """Formats the prompt from the chat messages."""

    def __init__(
        self,
        message_template: int = MESSAGE_TEMPLATE,
        system_prompt: int = SYSTEM_PROMPT,
    ):
        self.message_template = message_template
        self.messages = [{
            'role': 'System',
            'content': system_prompt
        }]

    def add_user_message(self, message: str) -> None:
        self.messages.append({
            'role': 'Human',
            'content': message
        })

    def add_bot_message(self, message: str) -> None:
        self.messages.append({
            'role': 'Assistant',
            'content': message
        })

    def get_prompt(self) -> str:
        final_text = ''
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        return final_text.strip()


class PhraseStoppingCriteria(StoppingCriteria):
    """Stops the model generation once the `target` phrase occurs."""

    def __init__(self, target: str, prompt: str, tokenizer: AutoTokenizer, **decode_kwargs) -> None:
        self.target = target
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.prompt_length = tokenizer.encode(prompt, return_tensors='pt').shape[1]
        self.decode_kwargs = decode_kwargs

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(
            input_ids[0][self.prompt_length:],
            **self.decode_kwargs
        )
        if self.target in generated_text:
            return True
        return False

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator:
        yield self


def stream_until(streamer: TextIteratorStreamer, phrase: str) -> Iterable[str]:
    """Helper function to exclude a `phrase` string from generation."""

    check = phrase.split(' ')[0]
    while True:
        try:
            token: str = next(streamer)
            if token.strip().endswith(check):
                token, sep, part = token.rpartition(check)
                part = sep + part
                while (
                    (part := part + next(streamer))
                    and len(part) <= len(phrase)
                    and part in phrase
                ):
                    if part == phrase:
                        yield token
                        return
                else:
                    token = token + part
            yield token
        except StopIteration:
            break


class OASSTGuanacoAdapter(LLMAdapterBase):

    STOP_PHRASE: str = STOP_PHRASE

    def __init__(self, llm: PreTrainedModel, model_id: str = MODEL_NAME) -> None:
        super().__init__(llm, model_id, adapter_name='oastt')

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample=True,
            top_k=0,
            top_p=1,
            temperature=0.25,
            repetition_penalty=1.2,
            max_new_tokens=250,
        )

    def preformat(self, message: str, history: list[tuple[str, str]]) -> str:
        chat = Conversation()
        for human, assistant in history:
            chat.add_user_message(human)
            chat.add_bot_message(assistant)
        chat.add_user_message(message)
        chat.add_bot_message('')
        return chat.get_prompt()

    def generator(self, llm: PreTrainedModel, prompt: str) -> Iterator[str]:
        data = self.tokenizer(prompt, return_tensors='pt')
        data = {k: v.to(llm.device) for k, v in data.items()}
        criteria = PhraseStoppingCriteria(
            STOP_PHRASE,
            prompt=prompt,
            tokenizer=self.tokenizer,
            skip_special_tokens=True
        )
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = dict(
            **data,
            generation_config=self.generation_config,
            streamer=streamer,
            stopping_criteria=criteria,
        )
        stream = Thread(target=llm.generate, kwargs=generation_kwargs)
        stream.start()
        return stream_until(streamer, STOP_PHRASE)
