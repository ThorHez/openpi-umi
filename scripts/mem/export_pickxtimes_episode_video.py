#!/usr/bin/env python3
"""Export an annotated front-camera video from the PickXtimes HDF5."""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import h5py
import numpy as np

EVENT_COLORS = {
    "pick_complete": (80, 220, 80),
    "place_complete": (255, 180, 60),
    "press_complete": (80, 100, 255),
}
EVENT_NAMES = {
    "pick_complete": "PICK complete",
    "place_complete": "PLACE complete",
    "press_complete": "PRESS complete",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def draw_text(image, text: str, origin: tuple[int, int], *, scale=0.7, color=(240, 240, 240), thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    metadata_by_index = {int(episode["episode_index"]): episode for episode in labels}
    if args.episode_index not in metadata_by_index:
        raise ValueError(f"Unknown episode index {args.episode_index}")
    metadata = metadata_by_index[args.episode_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    width = 768
    image_height = 768
    header_height = 192
    canvas_height = header_height + image_height
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    initial_state = {
        "completed_count": 0,
        "remaining_count": int(metadata["required_count"]),
        "holding": False,
        "should_press": False,
        "done": False,
    }
    anchors = [int(event["anchor"]) for event in metadata["events"]]
    num_steps = int(metadata["num_steps"])
    with h5py.File(args.h5, "r") as h5_file:
        episode = h5_file[metadata["episode_name"]]
        for frame_index in range(num_steps):
            rgb = episode[f"timestep_{frame_index}/obs/front_rgb"][()]
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (width, image_height), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((canvas_height, width, 3), 24, dtype=np.uint8)
            canvas[header_height:] = frame

            state = initial_state
            active_event = None
            for event in metadata["events"]:
                if int(event["start"]) <= frame_index <= int(event["end"]):
                    active_event = event
                if frame_index >= int(event["anchor"]):
                    state = event["state_after"]

            instruction = metadata["prompts"][0]
            midpoint = min(len(instruction), 82)
            if midpoint < len(instruction):
                split_at = instruction.rfind(" ", 0, midpoint)
                split_at = midpoint if split_at < 1 else split_at
                instruction_lines = (instruction[:split_at], instruction[split_at + 1 :])
            else:
                instruction_lines = (instruction, "")
            draw_text(
                canvas, f"PickXtimes episode {args.episode_index} | frame {frame_index}/{num_steps - 1}", (18, 28)
            )
            draw_text(canvas, instruction_lines[0], (18, 55), scale=0.48, thickness=1)
            if instruction_lines[1]:
                draw_text(canvas, instruction_lines[1], (18, 76), scale=0.48, thickness=1)

            event_text = "idle"
            event_color = (180, 180, 180)
            if active_event is not None:
                event_text = EVENT_NAMES[active_event["event_type"]]
                event_color = EVENT_COLORS[active_event["event_type"]]
                if abs(frame_index - int(active_event["anchor"])) <= 4:
                    event_text += "  [MEM EVENT WINDOW]"
                    cv2.rectangle(canvas, (3, header_height + 3), (width - 4, canvas_height - 4), event_color, 6)
            draw_text(canvas, f"current macro: {event_text}", (18, 105), color=event_color)
            draw_text(
                canvas,
                "memory target: "
                f"completed={state['completed_count']}  remaining={state['remaining_count']}  "
                f"holding={int(state['holding'])}  should_press={int(state['should_press'])}  "
                f"done={int(state['done'])}",
                (18, 133),
                scale=0.53,
                thickness=1,
            )

            line_left, line_right, line_y = 22, width - 22, 170
            cv2.line(canvas, (line_left, line_y), (line_right, line_y), (130, 130, 130), 2)
            for event_number, (event, anchor) in enumerate(zip(metadata["events"], anchors, strict=True), start=1):
                x = line_left + round(anchor / max(num_steps - 1, 1) * (line_right - line_left))
                color = EVENT_COLORS[event["event_type"]]
                cv2.circle(canvas, (x, line_y), 7, color, -1)
                short_name = {"pick_complete": "P", "place_complete": "L", "press_complete": "S"}[event["event_type"]]
                draw_text(
                    canvas, f"{short_name}{event_number}", (x - 10, line_y - 12), scale=0.35, color=color, thickness=1
                )
            cursor_x = line_left + round(frame_index / max(num_steps - 1, 1) * (line_right - line_left))
            cv2.line(canvas, (cursor_x, line_y - 16), (cursor_x, line_y + 13), (255, 255, 255), 2)
            writer.write(canvas)
    writer.release()
    print(f"Wrote {num_steps} frames to {args.output} at {args.fps:g} FPS")


if __name__ == "__main__":
    main()
