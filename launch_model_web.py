#!/usr/bin/env python3
"""Entrada da interface web HTMX do Crono launcher."""

import argparse
import ctypes
import os
import signal
import sys


def terminate_with_parent():
    """Ensure a launcher started in a terminal cannot survive that terminal."""
    if not sys.platform.startswith("linux"):
        return
    parent = os.getppid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            return
        # Close the race where the parent exits between getppid() and prctl().
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGTERM)
    except (OSError, AttributeError):
        pass


def main():
    parser = argparse.ArgumentParser(description="Crono Matrix web launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--keep-running", action="store_true",
        help="mantem o launcher ativo se o terminal que o iniciou for fechado",
    )
    parser.add_argument(
        "--agent-compat", dest="agent_compat",
        choices=("on", "off"), default=None,
        help=(
            "ativa/desativa o perfil universal para clientes OpenAI-compatible; "
            "gera arquivos apontando para o llama-server carregado"
        ),
    )
    args = parser.parse_args()
    if args.agent_compat is not None:
        # web.app importa o estado do launcher dentro do uvicorn. Definir o
        # ambiente antes do import permite que a opção da linha de comando
        # controle o padrão sem alterar a configuração de nenhum agente.
        os.environ["CRONO_AGENT_COMPAT"] = "y" if args.agent_compat == "on" else "n"
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Instale as dependencias: pip install -r requirements-web.txt") from exc
    if not args.keep_running:
        terminate_with_parent()
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
