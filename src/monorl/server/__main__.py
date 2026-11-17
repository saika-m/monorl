"""``python -m monorl.server [--host H] [--port P] [--open] [--token T] [--allowed-host NAME]``."""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="monorl.server", description="Run the monorl web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open", action="store_true", help="Open the UI in a browser once the server starts."
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--token", help="Access token (default: MONORL_SERVER_TOKEN, else a new random one).")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Accept this name in the Host header besides localhost, 127.0.0.1 and [::1].",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from monorl.core.errors import ConfigError
    from monorl.server import auth
    from monorl.server.app import create_app

    try:
        token = auth.configure(args.host, args.token, args.allowed_host)
    except ConfigError as e:
        parser.error(str(e))
    app = create_app()
    for line in auth.startup_lines(args.host, args.port, token):
        print(line, flush=True)
    if args.open:
        url = auth.signin_url(args.host, args.port, token)
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
