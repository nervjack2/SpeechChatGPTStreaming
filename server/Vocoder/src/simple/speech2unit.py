import torch
import joblib
import soundfile as sf
from transformers import Wav2Vec2Model
from argparse import ArgumentParser
from faster_whisper import WhisperModel
TPS = 50

def transcribe(audio):
    # Maybe we can use batch pipeline in faster whisper for better efficiency
    segments, info = ASR.transcribe(audio, beam_size=5, language="en", condition_on_previous_text=False, word_timestamps=True)
    return segments

def quantize(audio):
    with torch.no_grad():
        feats = HuBERT(torch.from_numpy(audio).unsqueeze(0).to(args.device).to(torch.float32 if not args.fp16 else torch.float16)).last_hidden_state
    pred = kmeans_model.predict(feats[0].float().cpu().numpy())
    return [f"<|{p}|>" for p in pred]
    

def combine(kms, segments):
    words = []
    for segment in segments:
        for w in segment.words:
            words.append((w.word, int(w.start * TPS)))
    for i, (w, s) in enumerate(words):
        kms.insert(i + s, ' ' + w)

    return ''.join(kms)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_audio", type=str, help="Input audio file")
    parser.add_argument("--output_path", type=str, default="tmp.txt", help="Path to save interleaving sequence")
    parser.add_argument("--device", type=str, default="cuda", help="Acceleration device")
    parser.add_argument("--kmeans_path", type=str, default="./hubert_ckpt/kmeans.bin")
    parser.add_argument("--fp16", action="store_true", help="Data types for quantizing HuBERT features. Using flash_attention_2 (float16), which is faster, but sometimes results in different results")
    args = parser.parse_args()

    # Initialize Whisper model for transcribing
    ASR = WhisperModel("distil-large-v3", device=args.device, compute_type="float16")

    # Initialize HuBERT model for feature extractoion
    if args.fp16:
        HuBERT = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(args.device)
    else:
        HuBERT = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float32).to(args.device)
    HuBERT.eval()

    # kmeans model for quantize HuBERT features
    kmeans_model = joblib.load(open(args.kmeans_path, "rb"))
    kmeans_model.verbose = False

    # Read audio
    audio, sr = sf.read(args.input_audio)
    assert sr == 16000, "Sample rate of audio should be 16000 Hz"
    
    # Transcribe given audio
    segments = transcribe(audio)

    # Quantize HuBERT features
    kms = quantize(audio)

    # Generate interleaving sequence
    interleave = combine(kms, segments)

    # Dump results
    with open(args.output_path, 'w') as f:
        f.write(interleave + '\n')
