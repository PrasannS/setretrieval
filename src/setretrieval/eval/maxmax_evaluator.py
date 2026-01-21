from __future__ import annotations
from sentence_transformers.evaluation import TripletEvaluator
from pylate.evaluation.colbert_triplet import csv_writer, evaluation_message
import os
from setretrieval.train.scores import maxmax_scores_pairwise
import logging
from contextlib import nullcontext
from pylate.models import ColBERT

logger = logging.getLogger(__name__)

class MaxMaxTripletEvaluator(TripletEvaluator):

    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ) -> None:
        super().__init__(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_distance_function=None,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
        )

        self.csv_headers = [
            "epoch",
            "steps",
            "accuracy",
        ]

        self.metrics = [
            "accuracy",
        ]

        self.primary_metric = "accuracy"

    def __call__(
        self,
        model: ColBERT,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        """Evaluate the model on the triplet dataset. Measure the scoring between the anchor
        and the positive with every other positive and negative samples using HITS@K.
        """
        evaluation_message(
            epoch=epoch, steps=steps, name=self.name, truncate_dim=self.truncate_dim
        )

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(truncate_dim=self.truncate_dim)
        ):
            embeddings_anchors = model.encode(
                sentences=self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                is_query=True,
            )
            embeddings_positives = model.encode(
                sentences=self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )
            embeddings_negatives = model.encode(
                sentences=self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )
        

        # Colbert pairwise scores
        positive_scores = maxmax_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_positives,
        )

        negative_scores = maxmax_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_negatives,
        )

        breakpoint()


        metrics = {
            "accuracy": (
                sum(positive_scores > negative_scores) / len(positive_scores)
            ).item()
        }

        for metric in self.metrics:
            logger.info(f"{metric.capitalize()}: \t{metrics[metric]:.2f}")

        self.store_metrics_in_model_card_data(model=model, metrics=metrics)

        if output_path is not None and self.write_csv:
            csv_writer(
                path=os.path.join(output_path, self.csv_file),
                header=self.csv_headers,
                data=[
                    epoch,
                    steps,
                ]
                + [metrics[metric] for metric in self.metrics],
            )

        return metrics
