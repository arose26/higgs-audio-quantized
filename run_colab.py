import os
from generation_two import load_model_and_tokenizer, main
import shutil
import re
from pydub import AudioSegment
import os 

 # Load model and tokenizer
model_client, audio_tokenizer = load_model_and_tokenizer(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer="bosonai/higgs-audio-v2-tokenizer",
    max_new_tokens=2048,
    device_id=None,
    use_static_kv_cache=1,
)


def concat_mp3s(input_filenames, output_filename='concat.wav'):
  # List your MP3 filenames here
  #mp3_files = ['file1.mp3', 'file2.mp3', 'file3.mp3']  # Replace with your files

  # Start with an empty audio segment
  combined = AudioSegment.empty()

  # Concatenate all wav files
  for input_file in input_filenames:
      audio = AudioSegment.from_wav(input_file)
      combined += audio

  # Export the result to 'concat.mp3'
  combined.export(output_filename, format='wav')


def process_text(text):
    """Process text to remove extra spaces and newlines"""
    #Remove [01:13:16] timestamps
    text = re.sub(r'\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]', '', text.strip())
    pairs = [('ok','OK'),("im","I'm"),("theyre","they're"),("wont","won't")]
    for a,b in pairs:
        text = text.replace(f" {a} ", f" {b} ")
    return text

def sanitize_filename(filename, replacement=""):
    # Remove invalid characters
    sanitized = re.sub(r'[^a-zA-Z]', '', filename)
    return sanitized[:30]


####
# For really long sentences (>350 chars), generate them independently then concat the mp3
###

def split_sentences(text):
    # Splits on period or ellipsis or question mark followed by space or end of string
    pattern = re.compile(r'(.*?(?:\.|\?)+)')
    sentences = pattern.findall(text)
    print(sentences)
    if not sentences:
        # Fallback: split on period
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    return sentences

def group_sentences(sentences, max_len=350):
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence + ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



def generate(input_text="Hi there.", index=None):
    filename = sanitize_filename(input_text) + '.wav'
    if index:
        filename = str(index) + '.' + filename


    # Get the directory where the current file resides
    current_file_dir = "/content/higgs-audio-quantized"#os.path.dirname(os.path.realpath(__file__))

    default_kwargs = dict(
        model_client=model_client,
        audio_tokenizer=audio_tokenizer,
        transcript="transcript/single_speaker/en_dl.txt",
        scene_prompt=f"/content/higgs-audio-quantized/scene_prompts/quiet_indoor.txt",
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        ref_audio=None,
        ref_audio_in_system_message=False,
        chunk_method=None,
        chunk_max_word_num=200,
        chunk_max_num_turns=1,
        generation_chunk_buffer_size=None,
        seed=None,
        out_path="generation.wav",
    )
    if len(input_text) <= 350:
        kwargs = default_kwargs.copy()
        kwargs.update(dict(transcript=input_text, ref_audio="trash", temperature=0.3,
                  out_path=filename))
        main(**kwargs)
        #files.download(filename)
        return

    #New - long passage support
    sentences = split_sentences(input_text)
    chunks = group_sentences(sentences, max_len=350)

    temp_files = []
    for i, chunk in enumerate(chunks):
        temp_filename = f"temp_{i}.wav"
        kwargs = default_kwargs.copy()
        kwargs.update(dict(transcript=chunk, ref_audio="trash", temperature=0.3,
                  out_path=temp_filename))
        main(**kwargs)
        temp_files.append(temp_filename)

    concat_mp3s(temp_files)

    if os.path.exists('concat.wav'):
        shutil.move('concat.wav', filename)
        #files.download(filename)
    else:
        print("Error: concat.wav not found")

    for f in temp_files:
        os.remove(f)



if __name__ == "__main__":
    if os.path.exists('/content/narration.txt'):
        with open('/content/narration.txt', 'r') as f:
            texts = [process_text(line) for line in f if line]
    else: #Attempt to get from gist
        texts = [process_text(line) for line in get_most_recent_gist().splitlines()]

    for i, text in enumerate(texts):
        generate(text, i+1)