from pathlib import Path
import signal
import time
from typing import NamedTuple
from multiprocessing import Queue
from multiprocessing import Process

import numpy
import pyaudio

from become_yukarin import AcousticConverter
from become_yukarin import RealtimeVocoder
from become_yukarin import SuperResolution
from become_yukarin import VoiceChanger
from become_yukarin.data_struct import Wave
from become_yukarin.voice_changer import VoiceChangerStream
from become_yukarin.voice_changer import VoiceChangerStreamWrapper

from become_yukarin.config.config import create_from_json as create_config_1st
from become_yukarin.config.sr_config import create_from_json as create_config_2nd

from numba import jit
import json

class AudioConfig(NamedTuple):
    rate: int
    audio_chunk: int
    convert_chunk: int
    vocoder_buffer_size: int
    out_norm: float

@jit
def convert_worker(
        config,
        acoustic_converter,
        super_resolution,
        audio_config: AudioConfig,
        queue_input_wave,
        queue_output_wave,
):
    vocoder = RealtimeVocoder(
        acoustic_feature_param=config.dataset.param.acoustic_feature_param,
        out_sampling_rate=audio_config.rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )
    # vocoder.warm_up(audio_config.vocoder_buffer_size / config.dataset.param.voice_param.sample_rate)

    voice_changer = VoiceChanger(
        super_resolution=super_resolution,
        acoustic_converter=acoustic_converter,
        vocoder=vocoder,
    )

    voice_changer_stream = VoiceChangerStream(
        voice_changer=voice_changer,
        sampling_rate=audio_config.rate,
        in_dtype=numpy.float32,
    )

    wrapper = VoiceChangerStreamWrapper(
        voice_changer_stream=voice_changer_stream,
        extra_time=0.1,
    )

    start_time = 0
    wave = numpy.zeros(audio_config.convert_chunk * 2, dtype=numpy.float32)
    wave = Wave(wave=wave, sampling_rate=audio_config.rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=wave)
    start_time += len(wave.wave) / wave.sampling_rate
    wave = wrapper.convert_next(time_length=1)

    time_length = audio_config.convert_chunk / audio_config.rate
    wave_fragment = numpy.empty(0)
    while True:
        wave = queue_input_wave.get()
        w = Wave(wave=wave, sampling_rate=audio_config.rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
        start_time += time_length

        b = time.time()
        wave = wrapper.convert_next(time_length=time_length).wave
        print('time', time.time()-b, flush=True)
        wrapper.remove_previous_wave()
        print('converted wave', len(wave), flush=True)

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        if len(wave_fragment) >= audio_config.audio_chunk:
            wave, wave_fragment = wave_fragment[:audio_config.audio_chunk], wave_fragment[audio_config.audio_chunk:]
            queue_output_wave.put(wave)


setting = json.load(open(Path("./script/setting.json")))

gpu = setting["gpu"]
CONFIG_1ST_PATH = Path(setting["model"]["1st_conf"])
MODEL_1ST_PATH  = Path(setting["model"]["1st_model"])
CONFIG_2ND_PATH = Path(setting["model"]["2nd_conf"])
MODEL_2ND_PATH  = Path(setting["model"]["2nd_model"])


def main():
    print('model loading...', flush=True)

    queue_input_wave = Queue()
    queue_output_wave = Queue()

    config_1st = create_config_1st(CONFIG_1ST_PATH)
    config_2nd = create_config_2nd(CONFIG_2ND_PATH)

    acoustic_converter = AcousticConverter(config_1st, MODEL_1ST_PATH, gpu=gpu)
    print('model 1 loaded!', flush=True)

    super_resolution = SuperResolution(config_2nd, MODEL_2ND_PATH, gpu=gpu)
    print('model 2 loaded!', flush=True)

    audio_instance = pyaudio.PyAudio()
    audio_config = AudioConfig(
        rate=config_1st.dataset.param.voice_param.sample_rate,
        audio_chunk=config_1st.dataset.param.voice_param.sample_rate,
        convert_chunk=config_1st.dataset.param.voice_param.sample_rate,
        vocoder_buffer_size=config_1st.dataset.param.voice_param.sample_rate // 16,
        out_norm=2.5,
    )

    process_converter = Process(target=convert_worker, kwargs=dict(
        config=config_1st,
        audio_config=audio_config,
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        queue_input_wave=queue_input_wave,
        queue_output_wave=queue_output_wave,
    ))
    process_converter.start()

    signal.signal(signal.SIGINT, lambda signum, frame: process_converter.terminate())

    audio_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.rate,
        frames_per_buffer=audio_config.audio_chunk,
        input=True,
        output=True,
    )

    while True:
        # input audio
        in_data = audio_stream.read(audio_config.audio_chunk)
        wave = numpy.fromstring(in_data, dtype=numpy.float32)
        print('input', len(wave), flush=True)
        queue_input_wave.put(wave)

        # output
        try:
            wave = queue_output_wave.get_nowait()
        except:
            wave = None

        if wave is not None:
            print('output', len(wave), flush=True)
            wave *= audio_config.out_norm
            b = wave.astype(numpy.float32).tobytes()
            audio_stream.write(b)


if __name__ == '__main__':
    main()
