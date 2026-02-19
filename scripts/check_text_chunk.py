#!/usr/bin/env python3
"""Check a text chunk and return corrected Quran text if detected."""

import argparse
import sys

from ijaza import LLMProcessor


def _read_text(cli_text: str | None) -> str:
    """Read text from --text or stdin."""
    if cli_text is not None:
        return cli_text.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    raise SystemExit("Provide text via --text or pipe input on stdin.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Return the input text unchanged when no Quran quote is detected, "
            "or return text with corrected ayah when detected."
        )
    )
    parser.add_argument(
        "--text",
        help="Input chunk to check. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also print detection details to stderr.",
    )
    args = parser.parse_args()

    text = _read_text(args.text)
    processor = LLMProcessor()
    result = processor.process(text)

    output_text = result.corrected_text if result.quotes else text
    print(output_text)

    if args.report:
        print(f"quotes_detected={len(result.quotes)}", file=sys.stderr)
        for quote in result.quotes:
            print(
                f"ref={quote.reference} confidence={quote.confidence:.3f} "
                f"corrected={quote.was_corrected}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
