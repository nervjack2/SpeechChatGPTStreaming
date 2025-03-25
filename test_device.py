import pyaudio

p = pyaudio.PyAudio()

# 打印所有音频设备的支持信息
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
    print(f"  Max Input Channels: {info['maxInputChannels']}")
    print(f"  Max Output Channels: {info['maxOutputChannels']}")
    
    # 检查支持的采样率
    supported_rates = []
    for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
        try:
            if info['maxOutputChannels'] > 0:  # 确保设备支持输出
                p.is_format_supported(rate,
                                      output_device=info['index'],
                                      output_channels=info['maxOutputChannels'],
                                      output_format=pyaudio.paInt16)
            if info['maxInputChannels'] > 0:  # 确保设备支持输入
                p.is_format_supported(rate,
                                      input_device=info['index'],
                                      input_channels=info['maxInputChannels'],
                                      input_format=pyaudio.paInt16)
            supported_rates.append(rate)
        except ValueError:
            pass
    print(f"  Supported Sample Rates: {supported_rates}")