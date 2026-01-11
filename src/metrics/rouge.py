import evaluate
from typing import Optional


def compute_rouge(predictions: list[str], actual: list[str]) -> Optional[dict[str, float]]:
    rouge = evaluate.load("rouge")

    return rouge.compute(
        predictions=predictions,
        references=actual,
        use_aggregator=True, 
        use_stemmer=True,
    )
