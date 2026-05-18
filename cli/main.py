from cli.config_loader import load_settings, mask_key
from cli.client import NexoidClient
from cli.ui import TerminalUI


def run():
    ui = TerminalUI()
    try:
        cfg = load_settings()
    except Exception as e:
        ui.error(str(e))
        return

    client = NexoidClient(cfg)
    ui.header(cfg.get("model", "atlas"), cfg["api_base_url"], cfg.get("streaming", True))
    print("Type /help for commands. /exit to quit.")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            print("bye")
            break
        if text == "/help":
            print("Commands: /help /exit /stream on|off /mode chat|inference /config")
            continue
        if text.startswith("/stream "):
            cfg["streaming"] = text.split()[-1].lower() == "on"
            print(f"streaming={cfg['streaming']}")
            continue
        if text.startswith("/mode "):
            cfg["mode"] = text.split()[-1].lower()
            print(f"mode={cfg['mode']}")
            continue
        if text == "/config":
            print({**cfg, "api_key": mask_key(cfg.get("api_key", ""))})
            continue

        ui.user(text)
        ui.set_state(ui.STATE_THINKING)
        if cfg.get("streaming", True):
            ui.set_state(ui.STATE_STREAMING)
            ui.bot_start()
            try:
                for tok in client.stream(text, cfg.get("temperature", 0.7)):
                    ui.bot_stream(tok)
                ui.bot_end()
                ui.set_state(ui.STATE_IDLE)
            except Exception:
                ui.set_state(ui.STATE_ERROR)
                ui.error("Streaming failed. Recovering with non-stream response.")
                ui.set_state(ui.STATE_RECOVERING)
                res = client.chat(text, cfg.get("temperature", 0.7)) if cfg.get("mode", "chat") == "chat" else client.inference(text, cfg.get("temperature", 0.7))
                ui.bot_start(); ui.bot_stream(str(res.get("data", res))); ui.bot_end(); ui.set_state(ui.STATE_IDLE)
        else:
            res = client.chat(text, cfg.get("temperature", 0.7)) if cfg.get("mode", "chat") == "chat" else client.inference(text, cfg.get("temperature", 0.7))
            ui.bot_start(); ui.bot_stream(str(res.get("data", res))); ui.bot_end(); ui.set_state(ui.STATE_IDLE)


if __name__ == "__main__":
    run()
