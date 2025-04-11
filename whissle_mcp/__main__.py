"""Main entry point for the Whissle MCP server."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def get_python_path():
    return sys.executable


def generate_config(auth_token: str | None = None):
    module_dir = Path(__file__).resolve().parent
    server_path = module_dir / "server.py"
    python_path = get_python_path()

    final_auth_token = auth_token or os.environ.get("WHISSLE_AUTH_TOKEN")
    if not final_auth_token:
        print("Error: Whissle auth token is required.")
        print("Please either:")
        print("  1. Pass the auth token using --auth-token argument, or")
        print("  2. Set the WHISSLE_AUTH_TOKEN environment variable, or")
        print("  3. Add WHISSLE_AUTH_TOKEN to your .env file")
        sys.exit(1)

    config = {
        "mcpServers": {
            "Whissle": {
                "command": python_path,
                "args": [
                    str(server_path),
                ],
                "env": {"WHISSLE_AUTH_TOKEN": final_auth_token},
            }
        }
    }

    return config 