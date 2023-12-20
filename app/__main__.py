"""A demo application showing how to load a shared LLM and change its behavior at runtime using LoRA adapters."""

import logging

from typing import Iterable

import gradio as gr

from transformers import AutoModelForCausalLM

from app.adapters import OASSTGuanacoAdapter
from app.adapters import SaigaAdapter
from app import config


log = logging.getLogger('app')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG if config.DEBUG else logging.INFO,
        format='%(levelname)s\t%(name)s: %(message)s'
    )

    llm = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        device_map=config.DEVICE
    )
    llm.eval()
    log.info('LLM loaded!')

    saiga = SaigaAdapter(llm)
    oastt = OASSTGuanacoAdapter(llm)

    adapters = {
        saiga.name: saiga,
        oastt.name: oastt,
    }
    model = oastt
    log.info('Adapters ready!')
    log.debug('Available adapters: %s', adapters.keys())

    def switch_adapter(adapter: str) -> None:
        global model
        model = adapters[adapter]
        log.info('Switched adapter to %r', adapter)

    def process_message(
        message: list[str],
        history: list[tuple[str, str]],
        adapter: str,
        clear: bool,
    ) -> Iterable[str]:

        log.debug('User: %r | Adapter: %r', message, adapter)

        if adapter != model.name:
            switch_adapter(adapter)
            if clear:
                history.clear()
                log.info('History cleaned!')

        output = ''
        for token in model.generate(message, history):
            output += token
            yield output

        log.debug('Assistant: %r', output)

    with gr.Blocks(title='LLM with LoRA Adapters') as app:
        dropdown = gr.Dropdown(
            choices=list(adapters.keys()),
            value=model.name,
            label='Adapter',
        )
        clearhist = gr.Checkbox(
            value=True,
            label='Clear history on switch',
            info='Whether to clear a chat history when switching adapters'
        )
        chat = gr.ChatInterface(
            process_message,
            additional_inputs=[dropdown, clearhist],
            description=__doc__,
        ).queue()

    app.launch()
