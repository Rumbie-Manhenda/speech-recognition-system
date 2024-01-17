import deepspeech
import contextlib
import wave 
import os
import glob
import time
import collections
import webrtcvad
import numpy as np
from jiwer import wer_default
import Levenshtein
from pydub import AudioSegment
import prettytable as pt

language = input("Select language (english/italian/spanish): ").lower()

def read_wave(path):
    if not path:
        print("Error: Empty file path.")
        return None, None, None

    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            if language != "english":
                # Generate silence
                silence_duration_ms = 1000
                silence_segment = AudioSegment.silent(duration=silence_duration_ms)

                # Read audio frames
                frames = wf.getnframes()
                pcm_data = wf.readframes(frames)

                # Convert the PCM data to AudioSegment
                audio = AudioSegment(
                    pcm_data,
                    frame_rate=wf.getframerate(),
                    sample_width=wf.getsampwidth(),
                    channels=wf.getnchannels()
                )

                # Add silence to the beginning of the audio
                audio_with_silence = silence_segment + audio

                # Convert the AudioSegment back to PCM data
                pcm_data_with_silence = audio_with_silence.raw_data

                duration = len(audio_with_silence) / 1000.0

                return pcm_data_with_silence, audio_with_silence.frame_rate, duration
            else:
                num_channels = wf.getnchannels()
                assert num_channels == 1
                sample_width = wf.getsampwidth()
                assert sample_width == 2
                sample_rate = wf.getframerate()
                assert sample_rate in (8000, 16000, 32000)
                frames = wf.getnframes()
                pcm_data = wf.readframes(frames)
                duration = frames / sample_rate
                return pcm_data, sample_rate, duration

    except FileNotFoundError:
        print(f"Error: File not found - {path}")
        return None, None, None


class Frame(object):
   """Represents a "frame" of audio data."""
   def __init__(self, bytes, timestamp, duration):
       self.bytes = bytes
       self.timestamp = timestamp
       self.duration = duration
 
def frame_generator(frame_duration_ms, audio, sample_rate):
   """
   Produces audio frames from PCM audio data.
   Accepts the target frame duration in milliseconds, the PCM data, and the sample rate.
   Generates frames with the specified duration.
   """
   n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
   offset = 0
   timestamp = 0.0
   duration = (float(n) / sample_rate) / 2.0
   while offset + n < len(audio):
       yield Frame(audio[offset:offset + n], timestamp, duration)
       timestamp += duration
       offset += n



def vad_collector(sample_rate, frame_duration_ms,
                 padding_duration_ms, vad, frames):
   """Excludes non-voiced audio frames and provides a generator that outputs PCM audio data.
    Parameters:
    sample_rate: The audio sample rate in Hz.
    frame_duration_ms: The duration of each frame in milliseconds.
    padding_duration_ms: The duration to pad the window, in milliseconds.
    vad: An instance of webrtcvad.Vad.
    frames: A source of audio frames (sequence or generator).
 
   """
   num_padding_frames = int(padding_duration_ms / frame_duration_ms)
  
   ring_buffer = collections.deque(maxlen=num_padding_frames)
  
   triggered = False
 
   voiced_frames = []
   for frame in frames:
       is_speech = vad.is_speech(frame.bytes, sample_rate)
 
       if not triggered:
           ring_buffer.append((frame, is_speech))
           num_voiced = len([f for f, speech in ring_buffer if speech])
           if num_voiced > 0.9 * ring_buffer.maxlen:
               triggered = True
               for f, s in ring_buffer:
                   voiced_frames.append(f)
               ring_buffer.clear()
       else:
         
           voiced_frames.append(frame)
           ring_buffer.append((frame, is_speech))
           num_unvoiced = len([f for f, speech in ring_buffer if not speech])
           if num_unvoiced > 0.9 * ring_buffer.maxlen:
               triggered = False
               yield b''.join([f.bytes for f in voiced_frames])
               ring_buffer.clear()
               voiced_frames = []

   if voiced_frames:
       yield b''.join([f.bytes for f in voiced_frames])


def apply_gain_amplification(audio, gain_db=10):
    audio = audio + gain_db
    audio = audio.apply_gain(gain_db)
    return audio

def apply_low_pass_filter(audio, cutoff_frequency=3000):
    audio = audio.low_pass_filter(cutoff_frequency)
    return audio



def vad_segment_generator(wavFile, aggressiveness):
   print("Caught the wav file @: %s" % (wavFile))
   audio, sample_rate, audio_length = read_wave(wavFile)
   assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
   vad = webrtcvad.Vad(int(aggressiveness))
   frames = frame_generator(30, audio, sample_rate)
   frames = list(frames)
   segments = vad_collector(sample_rate, 30, 300, vad, frames)
 
   # Apply gain and low-pass filter to each segment before VAD
   filtered_segments = []
   for segment in segments:
        audio_segment = AudioSegment(segment, frame_rate=sample_rate, sample_width=2, channels=1)
        audio_segment = apply_gain_amplification(audio_segment, gain_db=10)
        audio_segment = apply_low_pass_filter(audio_segment, cutoff_frequency=3000)
        pcm_data_filtered = audio_segment.raw_data
        filtered_segments.append(pcm_data_filtered)

   return filtered_segments, sample_rate, audio_length





language_models = {
    "english": "Models\deepspeech-0.9.3-models.pbmm",
    "italian": "Models\output_graph_it.pbmm",
    "spanish": "Models\output_graph_es.pbmm",
    
}

def select_language(language):
    if language in language_models:
        model_path = language_models[language]
    else:
        raise ValueError("Invalid language selection")

    return model_path


model_path = select_language(language)

def load_model(model_path, scorer):
    model_load_start = time.time()
    ds = deepspeech.Model(model_path)
    model_load_end = time.time() - model_load_start
    

    scorer_load_start = time.time()
    ds.enableExternalScorer(scorer)
    scorer_load_end = time.time() - scorer_load_start
    

    return [ds, model_load_end, scorer_load_end]


def resolve_models(dirName):
    pb_files = glob.glob(dirName + "\*.pbmm")
    if not pb_files:
        print("Error: No model files (.pbmm) found in the specified directory.")
        return None, None  
    else:
        pb = pb_files[0]
        print("Found Model: %s" % pb)
        
    
    scorer_files = glob.glob(dirName + "\*.scorer")
    if not scorer_files:
        print("No scorer files found.")
        return  None  
    else:
        scorer = scorer_files[0]
        print("Found scorer: %s" % scorer)
        return pb, scorer


'''

Run Inference on input audio file
@param ds: Deepspeech object
@param audio: Input audio for running inference on
@param fs: Sample rate of the input audio file
 
'''
def stt(ds, audio, fs):
   inference_time = 0.0
   audio_length = len(audio) * (1 / fs)
 
  
   print('Running inference...')
   inference_start =time.time()
   output = ds.stt(audio)
   inference_end =time.time()- inference_start
   inference_time += inference_end
   
 
   return [output, inference_time]
def calculate_wer(reference, hypothesis):
  
   
    reference = reference.lower()
    hypothesis = hypothesis.lower()

    
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

  
    num_words = len(reference_tokens)
    substitutions = 0
    deletions = 0
    insertions = 0

 
    alignment = Levenshtein.editops(reference_tokens, hypothesis_tokens)

    for operation, _, _ in alignment:
        if operation == 'replace':
            substitutions += 1
        elif operation == 'delete':
            deletions += 1
        elif operation == 'insert':
            insertions += 1

    
    wer = (substitutions + deletions + insertions) / num_words * 100

    return wer

def read_audio_files(folder_path):
    audio_files = []

    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            
            # Read the audio file using librosa
            audio, sr, duration = read_wave(file_path)

            # Append the audio data and sample rate to the list
            audio_files.append({
                "file_name": file,
                "audio_data": audio,
                "file_path": file_path
            })

    return audio_files
def generate_output_table(filenames, language, inference_times, wers,extensions):
    table = pt.PrettyTable()
    table.field_names = ['Filename', 'Language', 'Inference Time(s)', 'WER (%)']

    for filename,ext, language, inference_time, wer in zip(filenames, extensions,language, inference_times, wers):
        table.add_row([filename+ext, language, "{:.2f}s".format(inference_time), "{:.2f}%".format(wer)])

    print(table)

def main():
        # Define reference transcripts for English, Italian, and Spanish
   reference_transcripts = {
        "english": {
            "checkin.wav": "Where is the check-in desk?",
            "parents.wav": "I have lost my parents.",
            "suitcase.wav": "Please, I have lost my suitcase.",
            "what_time.wav": "What time is my plane?",
            "where.wav": "Where are the restaurants and shops?",
            "work.wav": "This is a lot of work.",
            "passport.wav": "I don't have my passport with me right now."
        },
        "italian": {
            "checkin_it.wav": "Dove e' il bancone?",
            "parents_it.wav": "Ho perso i miei genitori.",
            "suitcase_it.wav": "Per favore, ho perso la mia valigia.",
            "what_time_it.wav": "A che ora e’ il mio aereo?",
            "where_it.wav": "Dove sono i ristoranti e i negozi?",
        },
        "spanish": {
            "checkin_es.wav": "¿Dónde están los mostradores?",
            "parents_es.wav": "He perdido a mis padres.",
            "suitcase_es.wav": "Por favor, he perdido mi maleta.",
            "what_time_es.wav": "¿A qué hora es mi avión?",
            "where_es.wav": "¿Dónde están los restaurantes y las tiendas?",
        }
    }
  
   
   model = 'Models'
   dirName = os.path.expanduser(model)
   print(dirName)
 
   #audio = input('Enter the location of your file : ')
   if language== 'english':
        folder_path = "Ex4_audio\\EN\\"
        audio_files_data = read_audio_files(folder_path)
   elif language =="spanish":
        folder_path = "Ex4_audio\\ES\\"
        audio_files_data = read_audio_files(folder_path)
   elif language == "italian":
        folder_path = "Ex4_audio\\IT\\"
        audio_files_data = read_audio_files(folder_path)
       

# Accessing the data for each file
   
   aggressive = 1
    
    
   output_graph, scorer = resolve_models(dirName)
 
  
   model_path= select_language(language)
   model_retval = load_model(model_path, scorer)
 
   title_names = ['Filename','Language', 'Inference Time(s)', ' WER']
  # print("\n%-20s %-20s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4], title_names[5]))
    
   inference_time = 0.0
   filenames= []
   extensions=[]
   inference_times=[]
   wers= []
   for file_data in audio_files_data:
            waveFile = file_data['file_path']
            segments, sample_rate, audio_length = vad_segment_generator(waveFile, aggressive)
            f = open(waveFile.rstrip(".wav") + ".txt", 'w')
            print("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
   

            for i, segment in enumerate(segments):
                print("Processing chunk %002d" % (i,))
                audio = np.frombuffer(segment, dtype=np.int16)
                output = stt(model_retval[0], audio, sample_rate)
                inference_time += output[1]
                f.write(output[0] + " ")
                
                filename, ext = os.path.split(os.path.basename(waveFile))
                hypothesis_sentence = output[0]
                print(f"Hypothesis sentence:{hypothesis_sentence}")
                        
                reference_sentence = reference_transcripts[language].get(filename+ext, "")
                print(f"Reference statement: {reference_sentence}")

                if reference_sentence:
                        # Calculate WER for the current file
                        wer = calculate_wer(reference_sentence, hypothesis_sentence)
                        print(f"File: {filename+ext}, WER: {wer:.2f}%")
                else:
                        print(f"Reference sentence not found for {filename+ext}")
                wers.append(wer)
                filenames.append(filename)
                extensions.append(ext)
                inference_times.append(inference_time)
                
            f.close()
            generate_output_table(filenames, language, inference_times, wers,extensions)
if __name__ == '__main__':
   main()













