

# Common hallucination list of whisper
hallucination_list = [
    'A',
    'You',
    '不吝點贊訂閱轉發打賞支持明鏡與點點欄目',
    '贊訂閱轉發打賞支持明鏡與點點欄目',
    '幕由 Amara.org 社群提供',
    '轉發打賞支持明鏡與點點欄目',
    '字幕提供',
    '字幕由Amara.org社區提供',
    'Thank you.',
    ' you.',
    '-bye.',
    '啊',
    'IYou',
    '提供字幕',
    "I'm going to show you how to do it.",
    ' A, A,',
    '啊 啊 啊 啊',
    '的的的的的的的',
    'Thank you for watching!',
    'Okay',
    "I don't know.",
    "I don't have it.",
    "Yeah.",
    "中文字幕",
    "I'm sorry.",
    "Domin",
]

def check_hallucination(transcription):
    for x in hallucination_list:
        # 判斷是否與原字串相同
        if (transcription in x) or (x in transcription):
            print(f"Hallucination detected: {transcription}")
            return True  
        # 判斷是否與重複n次的字串相同
        n = int(len(transcription)//len(x))
        if x * n == transcription:
            print(f"Hallucination detected: {transcription}")
            return True
    return False