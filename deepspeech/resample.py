from pydub import AudioSegment

def preprocess_audio(input_path, output_path, target_sr=16000, target_width=2):
  
    audio = AudioSegment.from_file(input_path)
    # Resample the audio to the target sample rate
    audio = audio.set_frame_rate(target_sr)
    audio = audio.set_sample_width(target_width)
    audio.export(output_path, format="wav")


input_path = "Ex4_audio\EN\work1.wav"
output_path = "Ex4_audio\EN\work.wav"
target_sample_rate = 16000
target_sample_width = 2

preprocess_audio(input_path, output_path, target_sample_rate, target_sample_width)
