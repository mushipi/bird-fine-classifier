"""エントリポイント。各サブコマンドは `uv run python -m bird_fine.<module>` で呼ぶ。"""


def main():
    print("bird-fine-classifier: use submodules directly, e.g.:")
    print("  uv run python -m bird_fine.data.download")
    print("  uv run python -m bird_fine.data.preprocess")
    print("  uv run python -m bird_fine.training.train")


if __name__ == "__main__":
    main()
