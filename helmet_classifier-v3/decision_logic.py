from __future__ import annotations

from .config import AppConfig
from .constants import MANAGER_HELMET_COLORS, MANAGER_VEST_COLOR, WORKER_VEST_COLORS
from .schemas import Box, DecisionResult, HelmetColorResult, VestColorResult


def decide_person_label(
    *,
    helmet_box: Box | None,
    torso_box: Box | None,
    helmet_result: HelmetColorResult,
    vest_result: VestColorResult,
    config: AppConfig,
) -> DecisionResult:
    helmet_match = helmet_box is not None and helmet_result.helmet_color in MANAGER_HELMET_COLORS
    vest_match = torso_box is not None and vest_result.vest_color == MANAGER_VEST_COLOR
    worker_vest_match = torso_box is not None and vest_result.vest_color in WORKER_VEST_COLORS

    if not config.enable_joint_decision:
        is_legacy_worker = helmet_box is not None and helmet_result.helmet_color == "red"
        return DecisionResult(
            label="worker" if is_legacy_worker else "manager",
            helmet_match_manager_rule=helmet_match,
            vest_match_manager_rule=False,
            manager_rule_matched=False,
            final_decision_rule="legacy_head_only_red_rule" if is_legacy_worker else "legacy_head_only_non_red_rule",
        )

    helmet_unknown = helmet_box is None or helmet_result.helmet_color == "unknown"

    if helmet_match and not worker_vest_match:
        return DecisionResult(
            label="manager",
            helmet_match_manager_rule=True,
            vest_match_manager_rule=vest_match,
            manager_rule_matched=True,
            final_decision_rule="帽子主判命中管理色",
        )

    if helmet_match and worker_vest_match:
        return DecisionResult(
            label="worker",
            helmet_match_manager_rule=True,
            vest_match_manager_rule=False,
            manager_rule_matched=False,
            final_decision_rule="帽子命中管理色，但反光衣命中作业色，按辅助规则修正为工作人员",
        )

    if helmet_unknown and vest_match:
        return DecisionResult(
            label="manager",
            helmet_match_manager_rule=False,
            vest_match_manager_rule=True,
            manager_rule_matched=True,
            final_decision_rule="帽子缺失或未知，反光衣辅助兜底命中管理色",
        )

    return DecisionResult(
        label="worker",
        helmet_match_manager_rule=helmet_match,
        vest_match_manager_rule=vest_match,
        manager_rule_matched=False,
        final_decision_rule="帽子主判未命中管理色",
    )
