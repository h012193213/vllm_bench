#!/usr/bin/env python3
"""Stream chat completions from an OpenAI-compatible server."""

import argparse
import json
import sys

import requests

DEFAULT_URL = "http://59.127.5.172:11125/v1/chat/completions"
DEFAULT_MODEL = "QuantTrio/Qwen3.6-27B-AWQ"


def stream_chat(
    url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    *,
    enable_thinking: bool = True,
) -> None:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {"model": model, "messages": messages, "stream": True}
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue

            data = raw_line[6:]
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})

            if enable_thinking:
                text = delta.get("content") or delta.get("reasoning")
            else:
                text = delta.get("content")
            if text:
                print(text, end="", flush=True)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream chat from OpenAI-compatible API")
    parser.add_argument("prompt", nargs="?", default="Explain AI in one sentence.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", default="You are a helpful assistant.")
    cot_group = parser.add_mutually_exclusive_group()
    cot_group.add_argument(
        "--no-think",
        action="store_true",
        help="disable chain-of-thought / reasoning (Qwen3: enable_thinking=false)",
    )
    cot_group.add_argument(
        "--think",
        action="store_true",
        help="enable chain-of-thought / reasoning (default)",
    )
    args = parser.parse_args()

    try:
        stream_chat(
            args.url,
            args.model,
            args.prompt,
            args.system,
            enable_thinking=not args.no_think,
        )
    except requests.RequestException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
