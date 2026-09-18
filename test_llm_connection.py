from openai import OpenAI
import sys

from configs import AVAILABLE_LLMs, DEFAULT_LLM, PARSER_LLM


def check_client(name: str):
    cfg = AVAILABLE_LLMs[name]
    kwargs = {"api_key": cfg["api_key"]}
    if "base_url" in cfg:
        kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        temperature=0,
    )
    print(f"{name} -> {resp.choices[0].message.content!r}")


def main():
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            print(f"TEST_LLM={name}")
            check_client(name)
        return
    print(f"DEFAULT_LLM={DEFAULT_LLM}")
    print(f"PARSER_LLM={PARSER_LLM}")
    check_client(DEFAULT_LLM)
    if PARSER_LLM != DEFAULT_LLM:
        check_client(PARSER_LLM)


if __name__ == "__main__":
    main()
