"""第1回: 素のLLMに1回だけ質問するスクリプト（発展: 応答時間とトークン数計測）。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

環境変数 AIE_MODEL を変えることで、モデル別の応答時間とトークン数を比較可能:
    AIE_MODEL=gemini-2.5-flash uv run python -m app.ask
"""

import argparse
import time

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentResponseUsageMetadata

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def format_metrics(elapsed_seconds: float, usage: GenerateContentResponseUsageMetadata | None) -> str:
    """応答時間と利用トークン数の計測結果を整形する。"""
    lines = [
        "--------------------------------------------------",
        "[計測結果]",
        f"- 応答時間: {elapsed_seconds:.2f} 秒 ({elapsed_seconds * 1000:.0f} ms)",
    ]
    if usage:
        prompt = getattr(usage, "prompt_token_count", 0)
        candidates = getattr(usage, "candidates_token_count", 0)
        total = getattr(usage, "total_token_count", prompt + candidates)
        lines.append(f"- 入力トークン数: {prompt}")
        lines.append(f"- 出力トークン数: {candidates}")
        lines.append(f"- 合計トークン数: {total}")
    else:
        lines.append("- トークン情報: メタデータを取得できませんでした")
    lines.append("--------------------------------------------------")
    return "\n".join(lines)


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())

    start_time = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed_seconds = time.perf_counter() - start_time

    answer = response.text or ""
    print(f"[回答]\n{answer}\n")
    print(format_metrics(elapsed_seconds, response.usage_metadata))

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問し、応答時間とトークン数を計測する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        ask(args.question)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
