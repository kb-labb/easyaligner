import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

from easyaligner.data.datamodel import AudioMetadata, SpeechSegment
from easyaligner.vad.utils import drop_empty_speeches, encode_vad_segments


def load_vad_model(onnx=False, opset_version=16):
    """
    Load the Silero VAD model.

    Parameters
    ----------
    onnx : bool, default False
        Whether to load the ONNX version of the model.
    opset_version : int, default 16
        The opset version for the ONNX model.

    Returns
    -------
    object
        The loaded Silero VAD model.
    """
    return load_silero_vad(onnx=onnx, opset_version=opset_version)


def merge_chunks(segments, chunk_size=30):
    """
    Merge Silero VAD segments into larger chunks of a maximum size.

    Parameters
    ----------
    segments : list of dict
        List of dictionaries with 'start' and 'end' keys for each speech segment.
    chunk_size : int, default 30
        The maximum duration for each chunk in seconds.

    Returns
    -------
    list of dict
        List of merged chunks, where each chunk is a dictionary with
        "start", "end", and "segments" keys.
    """
    if not segments:
        return []

    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    merged_segments = []
    subsegments = []

    for segment in segments:
        if segment["end"] - current_start > chunk_size and current_end - current_start > 0:
            merged_segments.append(
                {"start": current_start, "end": current_end, "segments": subsegments}
            )
            current_start = segment["start"]
            subsegments = []
        current_end = segment["end"]
        subsegments.append((segment["start"], segment["end"]))

    merged_segments.append(
        {"start": current_start, "end": segments[-1]["end"], "segments": subsegments}
    )
    return merged_segments


def run_vad_pipeline(
    metadata: AudioMetadata, model, audio: torch.Tensor, sample_rate=16000, chunk_size=30
):
    """
    Run VAD pipeline on the given audio metadata.

    Parameters
    ----------
    metadata : AudioMetadata
        The audio metadata object to update with VAD results.
    model : object
        The loaded Silero VAD model.
    audio : torch.Tensor
        The audio signal.
    sample_rate : int, default 16000
        The sample rate of the audio.
    chunk_size : int, default 30
        The maximum chunk size in seconds.

    Returns
    -------
    AudioMetadata
        The updated metadata object.
    """

    if audio is None:
        return None

    if metadata.speeches is None:
        # Run VAD on entire audio
        vad_segments = get_speech_timestamps(
            audio,
            model,
            max_speech_duration_s=chunk_size,
            return_seconds=True,
        )

        vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
        segments = encode_vad_segments(vad_segments)

        # Create a single SpeechSegment based on where speech was detected.
        # An empty `speeches` list signals a file with no detected speech.
        metadata.speeches = []
        if segments:
            metadata.speeches.append(
                SpeechSegment(
                    start=segments[0].start, end=segments[-1].end, text=None, chunks=segments
                )
            )
    else:
        # Run VAD on each speech segment
        for speech in tqdm(metadata.speeches, desc="Running VAD on speeches"):
            start = int(speech.start * sample_rate) if speech.start is not None else None
            end = int(speech.end * sample_rate) if speech.end is not None else None
            # Note: Using `None` as a slicing parameter is the same as omitting it
            speech_audio = audio[start:end]
            vad_segments = get_speech_timestamps(
                speech_audio,
                model,
                max_speech_duration_s=chunk_size,
                return_seconds=True,
            )
            vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
            # Add speech.start offset to each segment
            offset = speech.start if speech.start is not None else 0
            vad_segments = [
                {
                    "start": seg["start"] + offset,
                    "end": seg["end"] + offset,
                    "segments": seg["segments"],
                }
                for seg in vad_segments
            ]
            segments = encode_vad_segments(vad_segments)

            if speech.duration is None and segments:
                speech.start = segments[0].start
                speech.end = segments[-1].end
                speech.calculate_duration()

            speech.chunks = segments

    return drop_empty_speeches(metadata)
