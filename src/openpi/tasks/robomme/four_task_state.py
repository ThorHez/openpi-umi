"""Pure-Python recurrent state contracts for the four-task RoboMME pilot.

These small state machines are deliberately independent of JAX and simulator
metadata.  Dataset builders can use clean labels or Qwen pseudo-events to
produce the same committed state trajectory that later supervises the visual
recurrent memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


COLORS = ("red", "green", "blue")


def _check_color(color: str) -> None:
    if color not in COLORS:
        raise ValueError(f"Unsupported color: {color!r}")


def _swap_cell(cell: str | None, cell_a: str, cell_b: str) -> str | None:
    if cell == cell_a:
        return cell_b
    if cell == cell_b:
        return cell_a
    return cell


@dataclass(frozen=True)
class TargetIdentityState:
    """Color-conditioned target locations for (Video)Unmask tasks."""

    target_colors: tuple[str, ...]
    target_cells: tuple[str | None, ...]
    covered: bool = False
    completed_swap_count: int = 0
    next_pick_rank: int = 0

    @classmethod
    def empty(cls, target_colors: Iterable[str]) -> "TargetIdentityState":
        colors = tuple(target_colors)
        if not 1 <= len(colors) <= 2:
            raise ValueError(f"Expected one or two target colors, got {colors!r}")
        for color in colors:
            _check_color(color)
        return cls(colors, (None,) * len(colors))

    def observe_target(self, color: str, cell: str, *, covered: bool) -> "TargetIdentityState":
        _check_color(color)
        if color not in self.target_colors:
            return self
        cells = list(self.target_cells)
        cells[self.target_colors.index(color)] = cell
        return TargetIdentityState(
            self.target_colors,
            tuple(cells),
            covered=covered or self.covered,
            completed_swap_count=self.completed_swap_count,
            next_pick_rank=self.next_pick_rank,
        )

    def apply_swap(self, cell_a: str, cell_b: str) -> "TargetIdentityState":
        if cell_a == cell_b:
            raise ValueError("A swap requires two distinct cells")
        return TargetIdentityState(
            self.target_colors,
            tuple(_swap_cell(cell, cell_a, cell_b) for cell in self.target_cells),
            covered=self.covered,
            completed_swap_count=self.completed_swap_count + 1,
            next_pick_rank=self.next_pick_rank,
        )

    def complete_pick(self) -> "TargetIdentityState":
        next_rank = min(self.next_pick_rank + 1, len(self.target_colors))
        return TargetIdentityState(
            self.target_colors,
            self.target_cells,
            covered=self.covered,
            completed_swap_count=self.completed_swap_count,
            next_pick_rank=next_rank,
        )


@dataclass(frozen=True)
class OrderedTargetState:
    """Ordered target slots used by VideoPlaceOrder."""

    target_color: str
    queried_ordinal: int
    target_cells: tuple[str | None, ...] = (None, None, None, None)
    written_count: int = 0
    completed_swap_count: int = 0

    def __post_init__(self) -> None:
        _check_color(self.target_color)
        if self.queried_ordinal not in (1, 2, 3, 4):
            raise ValueError(f"Invalid queried ordinal: {self.queried_ordinal}")
        if len(self.target_cells) != 4:
            raise ValueError("OrderedTargetState always reserves four target slots")

    def place_complete(self, ordinal: int, cell: str) -> "OrderedTargetState":
        if ordinal not in (1, 2, 3, 4):
            raise ValueError(f"Invalid placement ordinal: {ordinal}")
        if ordinal > self.written_count + 1:
            raise ValueError(
                f"Cannot write ordinal {ordinal} before ordinal {self.written_count + 1}"
            )
        cells = list(self.target_cells)
        cells[ordinal - 1] = cell
        return OrderedTargetState(
            self.target_color,
            self.queried_ordinal,
            tuple(cells),
            written_count=max(self.written_count, ordinal),
            completed_swap_count=self.completed_swap_count,
        )

    def swap_complete(self, cell_a: str, cell_b: str) -> "OrderedTargetState":
        if cell_a == cell_b:
            raise ValueError("A swap requires two distinct cells")
        return OrderedTargetState(
            self.target_color,
            self.queried_ordinal,
            tuple(_swap_cell(cell, cell_a, cell_b) for cell in self.target_cells),
            written_count=self.written_count,
            completed_swap_count=self.completed_swap_count + 1,
        )

    @property
    def queried_cell(self) -> str | None:
        return self.target_cells[self.queried_ordinal - 1]


@dataclass(frozen=True)
class PickCountState:
    """Online progress state for PickXtimes."""

    target_color: str
    required_count: int
    completed_count: int = 0
    holding: bool = False
    done: bool = False

    def __post_init__(self) -> None:
        _check_color(self.target_color)
        if self.required_count not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid required count: {self.required_count}")
        if not 0 <= self.completed_count <= self.required_count:
            raise ValueError(f"Invalid completed count: {self.completed_count}")

    @property
    def ready_to_press(self) -> bool:
        return self.completed_count == self.required_count and not self.holding and not self.done

    def apply(self, event: str) -> "PickCountState":
        if event == "pick_complete":
            if self.done or self.holding or self.completed_count >= self.required_count:
                raise ValueError(f"Illegal pick transition from {self!r}")
            return PickCountState(
                self.target_color, self.required_count, self.completed_count, holding=True
            )
        if event == "place_complete":
            if self.done or not self.holding:
                raise ValueError(f"Illegal place transition from {self!r}")
            return PickCountState(
                self.target_color,
                self.required_count,
                self.completed_count + 1,
                holding=False,
            )
        if event == "press_complete":
            if not self.ready_to_press:
                raise ValueError(f"Illegal press transition from {self!r}")
            return PickCountState(
                self.target_color,
                self.required_count,
                self.completed_count,
                holding=False,
                done=True,
            )
        if event in ("no_completed_event", "incomplete_event"):
            return self
        raise ValueError(f"Unsupported PickXtimes event: {event!r}")

