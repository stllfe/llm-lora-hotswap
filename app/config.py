"""App configuration."""

import os

from typing import Final


DEBUG: Final[bool] = bool(int(os.getenv('DEBUG', '0')))

BASE_MODEL: Final[str] = os.getenv('BASE_MODEL', 'TheBloke/Llama-2-7B-GPTQ')
DEVICE: Final[str] = os.getenv('DEVICE', 'cuda:0')
