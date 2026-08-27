from proactive_kv_cache.datasets import load_public_text_rows


def main() -> None:
    names = ["daily_dialog", "samsum", "ag_news", "dolly", "xsum"]
    for name in names:
        for mode in ("templated", "rag"):
            rows = load_public_text_rows(name, "train", 1, prompt_mode=mode)
            assert len(rows) == 1, (name, mode, len(rows))
            assert rows[0]["prompt"], (name, mode)
            print(f"PASS dataset={name} mode={mode}")


if __name__ == "__main__":
    main()
