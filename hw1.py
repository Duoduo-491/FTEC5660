#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once."""
    ### YOUR CODE HERE
    import os
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        max_tokens=8192,

    )

    return {"llm": llm}


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string."""
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = chain["llm"]

    system_text = "You are a precise receipt parser. Output only valid JSON."

    user_text = """Extract from this ONE supermarket receipt image.
Return ONLY JSON:
{
  "final_payment": number or null,
  "subtotal": number or null,
  "discounts": [{"label": "string", "amount": positive number}]
}

Rules:
- final_payment: the final amount paid after ROUNDING. Usually the last payment line (OCTOPUS, CARD, CASH, etc.). Do NOT use SUBTOTAL.
- subtotal: the printed SUBTOTAL (after discounts, before rounding).
- discounts: every discount / promotion / coupon / member / app / packaging-damage / percentage discount line. Output each amount as a POSITIVE number (e.g. -5.39 -> 5.39). Do NOT include ROUNDING.
- Do NOT sum across receipts. Do NOT compute totals.
- All amounts are HKD.
- Output JSON only, no markdown, no explanation.
"""

    def build_messages(path: Path):
        return [
            SystemMessage(content=system_text),
            HumanMessage(content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ]),
        ]

    def response_text(raw: Any) -> str:
        content = getattr(raw, "content", None)
        if content is None:
            content = str(raw)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            content = "\n".join(parts)
        if isinstance(content, str) and content.strip():
            return content
        # Fallback: if content is empty, try reasoning_content
        kwargs = getattr(raw, "additional_kwargs", None) or {}
        reasoning = kwargs.get("reasoning_content")
        if reasoning:
            return reasoning
        return content or ""



    def extract_json(text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Find the LAST {...} block; handles reasoning text with JSON at end.
        matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON found in: {text[:500]!r}")
        # Try from the longest candidate first
        candidates = sorted(matches, key=len, reverse=True)
        last_err = None
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception as e:
                last_err = e
                continue
        raise ValueError(f"Could not parse JSON. Last error: {last_err}")

    def to_decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        s = str(value).replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        return Decimal(s) if s else Decimal("0")

    all_messages = [build_messages(path) for path in images]
    raw_outputs = llm.batch(all_messages)

    total_paid = Decimal("0")
    total_without_discount = Decimal("0")

    for raw in raw_outputs:
        text = response_text(raw)
        print("========== TEXT ==========")
        print(text)
        print("==========================")
        data = extract_json(text)

        final_payment = to_decimal(data.get("final_payment"))
        subtotal = to_decimal(data.get("subtotal"))
        discounts = data.get("discounts") or []
        discount_sum = sum(
            (to_decimal(d.get("amount")) for d in discounts),
            Decimal("0"),
        )

        total_paid += final_payment
        total_without_discount += subtotal + discount_sum

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
