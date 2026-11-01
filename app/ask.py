"""第1回: 素のLLMと継続して会話できる対話型スクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問で開始

会話履歴を保持しながら、'exit' や 'quit' を入力するまで続けて質問できる。
"""

import argparse

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentResponseUsageMetadata

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def format_token_usage(usage: GenerateContentResponseUsageMetadata | None) -> str:
    """利用トークン数のメタデータを可視化文字列に整形する。"""
    if not usage:
        return "[トークン情報] 取得できませんでした"
    prompt = usage.prompt_token_count or 0
    candidates = usage.candidates_token_count or 0
    total = getattr(usage, "total_token_count", prompt + candidates)
    return (
        f"[トークン消費] 入力: {prompt} | 出力: {candidates} | 合計: {total}"
    )


def ask(question: str) -> str:
    """単発の質問を実行し、回答とトークン数を返す（後方互換用）。"""
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print("-" * 50)
    print(format_token_usage(response.usage_metadata))
    print("-" * 50)

    return response.text or ""


def chat_loop(initial_question: str | None = None) -> None:
    """会話履歴を保持したまま継続して質問できる対話セッションを実行する。"""
    # Google GenAI SDK の chats.create によりクライアント側で会話履歴（History）を安全に保持
    client = genai.Client(api_key=get_api_key())
    chat_session = client.chats.create(model=MODEL)

    current_question = initial_question

    print("=" * 60)
    print("対話モードを開始しました ('exit' または 'quit' で終了)")
    print("会話履歴が保持されるため、過去の文脈を踏まえて続けて質問できます。")
    print("=" * 60)

    while True:
        if current_question is None:
            try:
                user_input = input("\n[あなた] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n対話を終了します。")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("対話を終了します。")
                break

            current_question = user_input

        print(f"\n[質問] {current_question}\n")

        try:
            response = chat_session.send_message(current_question)
            print(f"[回答]\n{response.text}\n")
            print("-" * 50)
            print(format_token_usage(response.usage_metadata))
            print("-" * 50)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            break

        # 初回質問の処理が終わったら次からは対話入力に切り替え
        current_question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMと継続して対話する")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"最初の質問（省略時は '{DEFAULT_QUESTION}'）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    chat_loop(initial_question=args.question)


if __name__ == "__main__":
    main()
