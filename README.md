# LoRA Adapters Hotswap

Это демо приложение на базе [Gradio](https://www.gradio.app/), демонстрирующее возможности горячей замены PEFT-адаптеров, а именно LoRA, над одной и той же LLM прямо в Runtime. Выполнено в рамках тестового задания на позицию ML Engineer.

## Запуск

### Вручную
Создать окружение Python 3.10.10 с помощью Conda или Pyenv:
```shell
conda create -n myenv python=3.10.10 && conda activate myenv
```
Установить необходимые пакеты:
```shell
pip install -r requirements.txt
```

И запустить:
```shell
python -m app
```

### Docker
Собрать образ:
```shell
docker build -t llm-lora-hotswap .
```
И запустить приложение:
```shell
docker run --gpus all --name hotswap-app --net host --rm -it llm-lora-hotswap
```

После запуска приложение должно быть доступно на 7860 порту localhost (порт Gradio по умолчанию).

## Особенности

В данном проекте я набросал черновую структуру классов, которые могли бы быть прототипом для решения в продакшене. Поскольку у каждого адаптера могут быть свои нюансы токенизации, пре/пост-обработки сообщений, они инкапсулируются в классах-наследниках [`LLMAdapterBase`](app/adapters/base.py).

Для самих ответов LLM я реализовал поддержку стриминга токенов, чтобы можно было отдать первый токен как можно быстрее на целевой интерфейс (в данном случае — в UI чата).

## Компоненты

- [Llama2 7B GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ) — LLM, квантизованная с помощью метода GPTQ до 4bit. Выбрал её вместо квантизации через `bitsandbytes`, поскольку по тестам GPTQ даёт выше качество итоговой модели. На моей локальной машине установлена RTX 3070 на 8Gb VRAM, поэтому нужна была хотя бы 4bit версия
- [Saiga2 LoRA](https://huggingface.co/IlyaGusev/saiga2_7b_lora) — адаптер поверх Llama 2, дообученный на инструктивно-диалоговом датасете [Сайга](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga)
- [Llama 2 LoRA OpenAssistant Guanaco](https://huggingface.co/kaitchup/Llama-2-7B-oasstguanaco-adapter) ([блогпост](https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge)) — адаптер, дообученный на очищенной части датасета OpenAssistant (OASTT)

## В продакшене

Цель данного демо — показать с помощью простого кода реализацию горячей замены и предоставить интерактивный интерфейс для демонстрации работы. Если бы потребовалось реализовывать подобный функционал в виде REST API, я бы посмотрел такие решения как [OpenLLM](https://github.com/bentoml/OpenLLM). Согласно [вот этому обзору](https://sersavvov.com/blog/7-frameworks-for-serving-llms) фреймворков для инференса и текущей документации, OpenLLM — единственный, который поддерживает [адаптеры и их подмену в Runtime](https://github.com/bentoml/OpenLLM#%EF%B8%8F-serving-fine-tuning-layers).

Однако OpenLLM сам по себе, и в особенности с адаптерами, будет давать низкий RPS и высокий Latency. Дело в том, что при сёрвинге в продакшене кучи адаптеров страдает батчинг запросов — следует задуматься над более эффективной утилизацией GPU и формированием батчей. Я нашёл пару многообещающих решений для этой проблемы:
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)

Другой открытый вопрос — допустимо ли использовать адаптер, обученный поверх модели в половинной точности, с моделью квантизованной до 4bit. Допускаю, что может присутствовать деградация в качестве вывода такого адаптера. В продакшен разработке следовало бы проверить качество этой связки на downstream задачах.

Кроме того, при обучении своего адаптера под такой юзкейс я бы сразу смотрел в сторону quantization-aware методов:
- [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://huggingface.co/papers/2310.08659)
- [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2309.14717)
- [LQ-LoRA: Low-rank plus Quantized Matrix Decomposition for Efficient Language Model Finetuning](https://openreview.net/forum?id=xw29VvOMmU)

## С чем возникли сложности

### Базовая модель

Основную массу времени я потратил на поиск базовой модели и подходящих адаптеров. Согласно заданию, я нацелился на использование Saiga2 от Ильи Гусева, однако она в свою очередь является файнтюном над базовой Llama2, а не Chat/Instruct версией — что, как оказалось, редкость, если хочется использовать модель в чатботе. Большинство LoRA-адаптеров для чата файнтюнятся именно от Chat/Instruct-модели.

### API библиотеки PEFT

Также оказалась не совсем прозрачной логика методов `add_adapter()`, `set_adapter()`, `load_adapter()` из библиотеки [PEFT](https://huggingface.co/docs/peft/tutorial/peft_integrations#transformers). Так, к примеру, добавление адаптера с помощью конфига и метода `add_adapter()` не инициализирует сами веса адаптера и, судя по всему, нацелено именно на юзкейс файнтюнинга модели.

Для инференса же необходимо вызывать именно `load_adapter()` с указанием идентификатора модели-адаптера с хаба (или локальной папки). Чтобы разобраться с этим, пришлось посмотреть код соответвующих методов, поскольку документация на момент написания очень расплывчатая.

_Было забавно наблюдать, как подключенный адаптер Сайги не работает и базовая модель при виде русских символов в промпте выдаёт код, причём на C/C++ и под платформу Windows..._

### Вывод моделей

Кроме того, пришлось дописать некоторую логику по пост-процессингу вывода модели OASTT и попотеть над подбором гиперпараметров для генерации. Так модель очевидно плохо уловила специфику чата и старается продолжать реплики за человека. Поэтому я отлавливаю токены, соответствующие началу реплики `### Human:` и останавливаю генерацию на них.
