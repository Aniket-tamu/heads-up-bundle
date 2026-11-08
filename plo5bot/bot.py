# Extracted from screenshot: original bot.py lines 207-225.
# The preceding Bot class and imports are not included in the screenshot.

from . import exploit, strat_tuned  # noqa: E402,F401

DEFAULT = Bot(exploit, {"base": "plo5bot.strat_tuned"})


def new_half(config, memory):
    DEFAULT.new_half(config, memory)


def choose_action(obs, memory):
    return DEFAULT.choose_action(obs, memory)


def on_hand_end(result, memory):
    DEFAULT.on_hand_end(result, memory)


def warmup():
    DEFAULT.warmup()
