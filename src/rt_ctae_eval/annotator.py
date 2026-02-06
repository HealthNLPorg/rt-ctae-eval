import polars as pl
from collections.abc import Mapping, Container


def get_annotator_to_file_ids_mappping(
    annotator_to_file_ids_tsv: str,
) -> Mapping[str, Container[int]]:
    annotator_to_file_ids_df = pl.read_csv(annotator_to_file_ids_tsv, separator="\t")
    annotator_to_unique_file_ids = {}
    for annnotator_name, clustered_file_ids in zip(
        annotator_to_file_ids_df["annotator_name"], annotator_to_file_ids_df["file_ids"]
    ):
        annotator_to_unique_file_ids[annnotator_name] = set(
            map(int, clustered_file_ids.split(","))
        )
    return annotator_to_unique_file_ids


def get_id_to_annotator_mappping(
    annotator_ids_tsv: str, annotator_ids_to_ignore: Container[int]
) -> Mapping[int, str]:
    annotator_with_ids_df = pl.read_csv(annotator_ids_tsv, separator="\t")
    id_to_unique_annotator = {}
    for annnotator_name, clustered_ids in zip(
        annotator_with_ids_df["annotator_name"], annotator_with_ids_df["annotator_ids"]
    ):
        for annotator_id in map(int, clustered_ids.split(",")):
            if annotator_id not in annotator_ids_to_ignore:
                id_to_unique_annotator[annotator_id] = annnotator_name
    return id_to_unique_annotator
