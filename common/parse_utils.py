import json
from dataclasses import dataclass

import pandas as pd

from common.constants import ACTIONS_TO_REMOVE, ALL_ACTION_NAMES


@dataclass(frozen=True, order=True)
class BehaviorLabeled:
    agent: int
    target: int
    action: str
    orig_target: str

    def agent_str(self) -> str:
        return f"mouse{self.agent}"

    def to_str(self) -> str:
        return ",".join([self.agent_str(), str(self.orig_target), self.action])

    @staticmethod
    def from_str(beh: str):
        def norm(s):
            return s.replace("'", "").replace('"', "")

        agent, target, action = beh.split(",")
        agent = norm(agent)
        target = norm(target)
        action = norm(action)
        assert (
            action in ALL_ACTION_NAMES or action in ACTIONS_TO_REMOVE
        ), f"action: {action}, beh = {beh}"

        assert agent.startswith("mouse"), f"agent = {agent}, beh = {beh}"
        agent = int(agent.replace("mouse", ""))
        orig_target = str(target)
        if target == "self":
            target = agent
        else:
            assert target.startswith("mouse")
            target = int(target.replace("mouse", ""))

        return BehaviorLabeled(
            agent=agent, target=target, action=action, orig_target=orig_target
        )


def parse_behaviors_labeled(field: str | float) -> list[BehaviorLabeled]:
    if pd.isna(field):
        return []
    assert isinstance(field, str)
    items = json.loads(field)
    return list(set(BehaviorLabeled.from_str(item) for item in items))


def parse_behaviors_labeled_from_row(row: dict) -> list[BehaviorLabeled]:
    return parse_behaviors_labeled(row["behaviors_labeled"])


def behaviors_labeled_to_str(behaviors_labeled: list[BehaviorLabeled]) -> str:
    return json.dumps([beh.to_str() for beh in behaviors_labeled])
