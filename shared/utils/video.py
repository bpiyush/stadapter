import decord
import PIL.Image
import PIL
import numpy as np


def load_frame(video_path, index=0):
    vr = decord.VideoReader(video_path, num_threads=1)
    frame = PIL.Image.fromarray(vr[index].asnumpy())
    return frame


def load_first_and_final_frames(video_path):
    vr = decord.VideoReader(video_path, num_threads=1)
    frame_l = PIL.Image.fromarray(vr[0].asnumpy())
    frame_r = PIL.Image.fromarray(vr[-1].asnumpy())
    return [frame_l, frame_r]


def load_frames_linspace(video_path, st=None, et=None, n=8, num_threads=1, **vr_args):
    # decord.bridge.set_bridge('native')

    try:
        vr = decord.VideoReader(video_path, num_threads=num_threads, **vr_args)
    except Exception as e:
        print("Error loading video:", e, "for video:", video_path)
        # Return blank frames
        return [PIL.Image.new("RGB", (480, 256)) for _ in range(n)]

    fps = vr.get_avg_fps()
    if st is None:
        sf = 0
    else:
        sf = max(int(st * fps), 0)
    if et is None:
        ef = len(vr) - 1
    else:
        ef = min(int(et * fps), len(vr) - 1)
    if n == -1:
        n = ef - sf + 1
    indices = np.linspace(sf, ef, n, endpoint=True).astype(int)
    # convert_to_PIL = lambda x: PIL.Image.fromarray(x.asnumpy())

    try:
        frames = [PIL.Image.fromarray(vr[i].asnumpy()) for i in indices]
    except Exception as e:
        print("Error loading frames:", e, "for video:", video_path)
        # Return blank frames
        frames = [PIL.Image.new("RGB", (480, 256)) for _ in range(n)]

    # Close the video reader
    del vr

    return frames


def load_frames_linspace_with_first_and_last(video_path, n=8):
    """Loads n frames from a video, including the first and last frames."""
    assert n > 1, "n should be greater than 1"
    vr = decord.VideoReader(video_path, num_threads=1)
    indices = np.linspace(0, len(vr) - 1, n - 2).astype(int)
    frames = [PIL.Image.fromarray(vr[0].asnumpy())]
    frames += [PIL.Image.fromarray(x) for x in vr.get_batch(indices).asnumpy()]
    frames += [PIL.Image.fromarray(vr[-1].asnumpy())]
    return frames


def get_duration(path, return_fps=False):
    vr = decord.VideoReader(path, num_threads=1)
    if not return_fps:
        return len(vr) / vr.get_avg_fps()
    else:
        return len(vr) / vr.get_avg_fps(), vr.get_avg_fps()


def load_frames_at_timestamps(video_path, timestamps):
    """
    Loads frames at given timestamps from a video.

    Args:
        video_path (str): Path to the video file.
        timestamps (list): List of timestamps at which to load frames.
    """
    vr = decord.VideoReader(video_path, num_threads=1)
    duration = len(vr) / vr.get_avg_fps()
    assert max(timestamps) <= duration, \
        "Timestamps should be within the duration of the video."
    indices = [int(t * vr.get_avg_fps()) for t in timestamps]
    frames = [PIL.Image.fromarray(vr[i].asnumpy()) for i in indices]
    return frames
