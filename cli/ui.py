import sys
import time

RESET = "\033[0m"
BOLD = "\033[1m"
FG_CYAN = "\033[96m"
FG_GREEN = "\033[92m"
FG_RED = "\033[91m"
FG_GRAY = "\033[90m"
BG_BLACK = "\033[40m"


class TerminalUI:
    STATE_IDLE = "IDLE"
    STATE_THINKING = "THINKING"
    STATE_STREAMING = "STREAMING"
    STATE_ERROR = "ERROR"
    STATE_RECOVERING = "RECOVERING"

    def __init__(self):
        self.state = self.STATE_IDLE

    def header(self, model: str, base_url: str, streaming: bool):
        print(BG_BLACK + FG_CYAN + BOLD + "NEXOID TERMINAL" + RESET)
        print(FG_GRAY + f"model={model}  endpoint={base_url}  streaming={streaming}" + RESET)
        print("-" * 40)

    def set_state(self, s: str):
        self.state = s
        print(FG_GRAY + f"[{s}]" + RESET)

    def user(self, text: str):
        print(FG_GREEN + "user: " + RESET + text)

    def bot_start(self):
        sys.stdout.write(FG_CYAN + "nexoid: " + RESET)
        sys.stdout.flush()

    def bot_stream(self, token: str):
        sys.stdout.write(token)
        sys.stdout.flush()
        time.sleep(0.01)

    def bot_end(self):
        print()

    def error(self, msg: str):
        print(FG_RED + "error: " + RESET + msg)
