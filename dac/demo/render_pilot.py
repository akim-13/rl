"""Render four matched CartPole rollouts using Gymnasium states and Pillow.

Retrains pilot seed 0 for each of the four controllers. After each block in
SHOWN_BLOCKS, A's greedy policy plays one game from the first evaluation start
state. The games are drawn side by side above each controller's pilot mean
learning curve and piped to ffmpeg.
"""
import csv
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

from dac.controllers import make_controller
from dac.inner import BlockResult, InnerConfig, InnerRun

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
VIDEO_PATH = HERE / "cartpole-four-controllers.mp4"
FONT = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"

CONTROLLER_NAMES = ["noop", "random", "linear_eps", "lr_gate"]
TITLES = ["Fixed settings", "Random adjustments", "Linear exploration decay", "Score-gated learning rate"]
COLOURS = ["#83c5ff", "#ffad77", "#d9a5fc", "#70dfac"]
SHOWN_BLOCKS = [0, 10, 20, 30]

# Pilot settings, which the captions and the chart assume.
LAST_BLOCK = 30
STEPS_PER_BLOCK = 1000
TIME_LIMIT_STEPS = 500

# 25 fps at 4 game steps per frame plays at twice CartPole's real 50 steps per second.
FRAMES_PER_SECOND = 25
GAME_STEPS_PER_FRAME = 4
FRAMES_BEFORE_PLAY = 18
FRAMES_AFTER_PLAY = 27

# Layout, in pixels.
VIDEO_SIZE = (1080, 720)
BACKGROUND = "#101921"
TEXT_COLOUR = "#edf3f6"
PANEL_COLOUR = "#1d2a35"
TRACK_COLOUR = "#83929c"
CART_COLOUR = "#dce3e6"
GRID_COLOUR = "#384650"
PANEL_WIDTH, PANEL_HEIGHT = 514, 202
PIXELS_PER_POSITION_UNIT = 90
POLE_LENGTH = 48
CHART_LEFT, CHART_BOTTOM, CHART_WIDTH, CHART_HEIGHT = 73, 688, 955, 115
CHART_TOP_SCORE = 500
FONT_SIZES = [16, 20, 24, 28]


@dataclass
class Clip:
    """One greedy game, played after a shown block."""

    cart_states: list[np.ndarray]  # observation after reset, then after every step
    result: BlockResult            # block the game was played after


@dataclass
class ControllerRollouts:
    clips: dict[int, Clip]     # shown block -> its game
    mean_curve: list[float]    # pilot mean eval score over all seeds, blocks 0..LAST_BLOCK


def collect(name: str) -> ControllerRollouts:
    """Retrain seed 0 as in the pilot, playing a greedy game after each shown block."""
    metadata = json.loads((RESULTS / f"p2_{name}.json").read_text())
    config = InnerConfig(**metadata["config"])
    run = InnerRun(0, config)
    controller = make_controller(name, config, 0, **metadata["controller_args"])
    game_environment = gym.make(config.env_id)
    clips = {}
    with (RESULTS / f"p2_{name}.csv").open() as file:
        pilot_rows = list(csv.DictReader(file))
    mean_curve = _mean_score_per_block(pilot_rows)
    try:
        for block in range(LAST_BLOCK + 1):
            result = _next_result(run, controller, block)
            # The rerun must match the pilot's seed 0, otherwise the video shows a different run.
            recorded_score = _seed_zero_score(pilot_rows, block)
            if not np.isclose(result.eval_score, recorded_score):
                raise ValueError((name, block, result.eval_score, recorded_score))
            if block in SHOWN_BLOCKS:
                cart_states = _play_greedy_game(run, game_environment, config.eval_seed)
                clips[block] = Clip(cart_states, result)
    finally:
        run.close()
        game_environment.close()
    return ControllerRollouts(clips, mean_curve)


def _mean_score_per_block(pilot_rows: list[dict]) -> list[float]:
    mean_curve = []
    for block in range(LAST_BLOCK + 1):
        scores_of_block = [float(row["eval_score"]) for row in pilot_rows if int(row["block"]) == block]
        mean_curve.append(np.mean(scores_of_block))
    return mean_curve


def _seed_zero_score(pilot_rows: list[dict], block: int) -> float:
    return next(
        float(row["eval_score"])
        for row in pilot_rows
        if int(row["seed"]) == 0 and int(row["block"]) == block
    )


def _next_result(run: InnerRun, controller, block: int) -> BlockResult:
    """Block 0 is the untrained network; later blocks train with the controller's multipliers."""
    if block == 0:
        return run.history[0]
    lr_multiplier, epsilon_multiplier = controller(run)
    return run.run_block(lr_multiplier, epsilon_multiplier)


def _play_greedy_game(run: InnerRun, environment: gym.Env, start_seed: int) -> list[np.ndarray]:
    observation, _ = environment.reset(seed=start_seed)
    cart_states = [observation]
    game_ended = False
    while not game_ended:
        action = int(run.agent.greedy_actions(observation[None])[0])
        observation, _, terminated, truncated, _ = environment.step(action)
        cart_states.append(observation)
        game_ended = terminated or truncated
    return cart_states


class _Canvas:
    def __init__(self, fonts: dict[int, ImageFont.FreeTypeFont]) -> None:
        self.image = Image.new("RGB", VIDEO_SIZE, BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = fonts

    def write(self, x, y, message: str, size: int = 20, colour: str = TEXT_COLOUR) -> None:
        self.draw.text((x, y), message, font=self.fonts[size], fill=colour)


def _draw_frame(
    all_rollouts: list[ControllerRollouts], block: int, game_step: int, fonts: dict
) -> Image.Image:
    canvas = _Canvas(fonts)
    _draw_headings(canvas, block)
    for panel, rollouts in enumerate(all_rollouts):
        _draw_game_panel(canvas, panel, rollouts.clips[block], game_step)
    _draw_learning_curves(canvas, all_rollouts, block)
    return canvas.image


def _draw_headings(canvas: _Canvas, block: int) -> None:
    canvas.write(24, 12, "CartPole   /   Four training controllers", 28)
    canvas.write(
        24, 52,
        f"Block {block} of {LAST_BLOCK}   |   {block * STEPS_PER_BLOCK:,} training steps   |   "
        f"Same start, greedy play at 2× speed",
        20,
    )


def _draw_game_panel(canvas: _Canvas, panel: int, clip: Clip, game_step: int) -> None:
    """Cart and pole at game_step, held on the last state once the game is over."""
    column, row = panel % 2, panel // 2
    left, top = 18 + column * 530, 95 + row * 216
    colour = COLOURS[panel]
    canvas.draw.rounded_rectangle((left, top, left + PANEL_WIDTH, top + PANEL_HEIGHT), radius=8, fill=PANEL_COLOUR)

    last_step = len(clip.cart_states) - 1
    shown_step = min(game_step, last_step)
    cart_position, _, pole_angle, _ = clip.cart_states[shown_step]
    cart_x = left + 257 + float(cart_position) * PIXELS_PER_POSITION_UNIT
    cart_y = top + 103
    pole_top_x = cart_x + POLE_LENGTH * math.sin(pole_angle)
    pole_top_y = cart_y - POLE_LENGTH * math.cos(pole_angle)
    canvas.draw.line((left + 10, cart_y + 11, left + 504, cart_y + 11), fill=TRACK_COLOUR, width=2)
    canvas.draw.rectangle((cart_x - 17, cart_y - 3, cart_x + 17, cart_y + 10), fill=CART_COLOUR)
    canvas.draw.line((cart_x, cart_y, pole_top_x, pole_top_y), fill=colour, width=6)

    result = clip.result
    canvas.write(left + 12, top + 5, TITLES[panel], 24, colour)
    canvas.write(left + 12, top + 122, f"This game: {shown_step} steps{_game_status(shown_step, last_step)}", 20)
    canvas.write(
        left + 12, top + 158,
        f"10-game mean {result.eval_score:.1f}   |   LR {result.lr:.3g}   |   ε {result.epsilon:.3g}",
        16,
    )


def _game_status(shown_step: int, last_step: int) -> str:
    if shown_step != last_step:
        return ""
    if shown_step == TIME_LIMIT_STEPS:
        return "  ·  time limit"
    return "  ·  ended"


def _draw_learning_curves(canvas: _Canvas, all_rollouts: list[ControllerRollouts], block: int) -> None:
    """Each controller's pilot mean learning curve, up to the current block."""
    canvas.write(24, 532, "Pilot mean evaluation score across 30 training seeds", 20)
    for score in [0, 250, 500]:
        line_y = _chart_y(score)
        canvas.draw.line((CHART_LEFT, line_y, CHART_LEFT + CHART_WIDTH, line_y), fill=GRID_COLOUR)
        canvas.write(24, line_y - 10, str(score), 16)
    for panel, rollouts in enumerate(all_rollouts):
        points = [(_chart_x(shown), _chart_y(rollouts.mean_curve[shown])) for shown in range(block + 1)]
        if len(points) > 1:
            canvas.draw.line(points, fill=COLOURS[panel], width=3)
        end_x, end_y = points[-1]
        canvas.draw.ellipse((end_x - 3, end_y - 3, end_x + 3, end_y + 3), fill=COLOURS[panel])
    canvas.write(CHART_LEFT, 690, "Block 0", 16)
    canvas.write(CHART_LEFT + CHART_WIDTH - 74, 690, f"Block {LAST_BLOCK}", 16)


def _chart_x(block: int) -> float:
    return CHART_LEFT + block * CHART_WIDTH / LAST_BLOCK


def _chart_y(score: float) -> float:
    return CHART_BOTTOM - score * CHART_HEIGHT / CHART_TOP_SCORE


def _frames_of_block(all_rollouts: list[ControllerRollouts], block: int, fonts: dict):
    """Hold on the start, play the games, then hold on the end positions."""
    longest_game = max(len(rollouts.clips[block].cart_states) - 1 for rollouts in all_rollouts)
    frame_count = math.ceil(longest_game / GAME_STEPS_PER_FRAME) + FRAMES_BEFORE_PLAY + FRAMES_AFTER_PLAY
    for frame in range(frame_count):
        game_step = min(longest_game, max(0, (frame - FRAMES_BEFORE_PLAY) * GAME_STEPS_PER_FRAME))
        yield _draw_frame(all_rollouts, block, game_step, fonts)


def _ffmpeg_command(target: Path) -> list[str]:
    width, height = VIDEO_SIZE
    return [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FRAMES_PER_SECOND), "-i", "-",
        "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(target),
    ]


def main():
    with ProcessPoolExecutor(4) as pool:
        all_rollouts = list(pool.map(collect, CONTROLLER_NAMES))
    fonts = {size: ImageFont.truetype(str(FONT), size) for size in FONT_SIZES}
    with subprocess.Popen(_ffmpeg_command(VIDEO_PATH), stdin=subprocess.PIPE) as encoder:
        for block in SHOWN_BLOCKS:
            for frame in _frames_of_block(all_rollouts, block, fonts):
                encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("video encoding failed")


if __name__ == "__main__":
    main()
