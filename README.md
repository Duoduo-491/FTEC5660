# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
### Chain design

```mermaid
flowchart LR
  A[Receipt images] --> B[Vision LLM per image]
  B --> C[Structured JSON: final_payment, subtotal, discounts]
  C --> D[Python per-receipt calculation]
  D --> E[Python sum across receipts]
  E --> F[Return two HK$ answers]
```

###Description
I use deepseek-v4-flash-vision-exp as the vision backbone and LangChain to read
each supermarket receipt. For every image, the model extracts a structured JSON
object with final_payment, subtotal, and a list of discounts (all positive
numbers, rounding excluded). The LLM only parses the receipt; all arithmetic is
done in Python with Decimal.

Query 1 sums final_payment across receipts.
Query 2 sums subtotal + sum(discounts) across receipts, without adding back the
rounding line.

This follows the prompt-chaining idea from Tutorial 1: let the model read, let
code compute, and validate each intermediate result.
