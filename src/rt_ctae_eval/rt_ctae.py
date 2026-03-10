import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lseval.correctness_matrix import Correctness, CorrectnessMatrix
from lseval.datatypes import DocTimeRel, Entity, Relation

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
RT_CUI = "C1522449"


def cuis_are_radiation_treatment(cuis: Collection[str]) -> bool:
    return RT_CUI in cuis and len(cuis) == 1


def cuis_are_adverse_event(cuis: Collection[str]) -> bool:
    return RT_CUI not in cuis


class NaranjoScale(Enum):
    DOUBTFUL = "Doubtful"
    POSSIBLE = "Possible"
    PROBABLE = "Probable"
    CERTAIN = "Certain"
    NA = "N/A"

    @classmethod
    def _missing_(cls, value):
        return NaranjoScale.NA


class EventType(Enum):
    RTEntity = "Radiotherapy Treatment"
    AdverseEventEntity = "Adverse Event"
    NA = "N/A"

    @classmethod
    def _missing_(cls, value):
        return EventType.NA


@dataclass(eq=True, frozen=True)
class RTEntity(Entity):
    def __post_init__(self):
        if not cuis_are_radiation_treatment(self.cuis):
            logger.warning("%s is not a proper RT entity", self.label_studio_id)


@dataclass(eq=True, frozen=True)
class AdverseEventEntity(Entity):
    def __post_init__(self):
        if not cuis_are_adverse_event(self.cuis):
            logger.warning(
                "%s is not a proper adverse event entity", self.label_studio_id
            )


@dataclass(eq=True, frozen=True)
class CausalRelation(Relation):
    def __post_init__(self):
        if not isinstance(self.arg1, RTEntity):
            logger.warning(
                "%s is not a radiation treatment - convention is radiation treatment is the anchor",
                self.arg1.label_studio_id,
            )
        if not isinstance(self.arg2, AdverseEventEntity):
            logger.warning(
                "%s is not a adverse event - convention is adverse event is the anchor",
                self.arg2.label_studio_id,
            )
        if len(self.label) != 1:
            raise ValueError(
                f"Only supporting single labels currently, not {self.label}"
            )
        if not all(map(CausalRelation.validate_naranjo_label, self.label)):
            raise ValueError(f"Invalid causality label/s {self.label}")

    @staticmethod
    def validate_naranjo_label(label: Any) -> bool:
        return label in NaranjoScale


@dataclass
class AnnotatedFileScores:
    file_id: int
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity]
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity]
    causal_relation_correctness_matrix: CorrectnessMatrix[CausalRelation]
    dtr_correctness_matrices: Mapping[DocTimeRel, CorrectnessMatrix[Entity]] = field(
        default_factory=dict
    )
    cui_correctness_totals: Mapping[str, Mapping[Correctness, int]] = field(
        default_factory=dict
    )


@dataclass
class AnnotatedCorpusScores:
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity]
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity]
    causal_relation_correctness_matrix: CorrectnessMatrix[CausalRelation]
    dtr_correctness_matrices: Mapping[DocTimeRel, CorrectnessMatrix[Entity]] = field(
        default_factory=dict
    )
    cui_correctness_totals: Mapping[str, Mapping[Correctness, int]] = field(
        default_factory=dict
    )
