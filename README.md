# paddlespeech_forme
paddlespeech局部定制版，流式ASR与TTS可接受和输出8k音频

对语音流降采样与上采样（pdspeech定制）
非root用户pip默认安装位置：/home/yourname/.local/lib/python3.8/site-packages/paddlespeech

## 首先是TTS，
字节流从24k降低到8k便于freeswitch播放，需要修改 server/engine/tts/online/onnx/tts_engine.py     默认用onnx，所以需要改onnx目录下的文件。定位到末尾的run()函数。
一个简单易行（但效果不佳）的方法如下：

```python
# wav type: <class 'numpy.ndarray'>  float32, convert to pcm (base64)
wav = float2pcm(wav)  # float32 to int16
wav_bytes = wav.tobytes()  # to bytes

new = bytearray()   # 在字节流层面进行简单抽样。先构造bytearray()对象
for n, ele in enumerate(wav_bytes):   # 进行3到1的降采样。
    if n % 6 < 2:       # 这是每6个字节保留前2个。
        new.append(ele)

wav_base64 = base64.b64encode(new).decode('utf8')  # 改为new转换为 base64 
```

上述方式生成的语音流，存在刺啦刺啦的噪声。更高级的降采样，可以用scipy.signal.decimate专用函数。该函数执行抗混叠处理和下采样（抗混叠处理是为了消除以固定间隔从连续数据中采样时声音失真的处理）。重新采样的方式与减少数据点的方式类似。 而且对代码的改动更小，只需要添加两行，改动一行。
```python
from scipy.signal import decimate  # 在代码头部引入  【添加】
# 针对推理的wav进行迭代：
# wav type: <class 'numpy.ndarray'>  float32, convert to pcm (base64)
downwav = decimate(wav.flatten(),3).reshape(-1,1)  #【添加】 抗混叠滤波进行3:1的降采样。
wav = float2pcm(downwav)  # float32 to int16     【括号里面改为downwav】
wav_bytes = wav.tobytes()  # to bytes  ----------本行无改动
```

## ASR部分
输入语音从8k字节流提升到16k才能送入模型识别，修改 paddlespeech/server/engine/asr/online/python/asr_engine.py
而asr conf里面默认是Pythonasr_online，就修改python目录下的文件。
定位到210行左右，修改  extract_feat(self, samples: ByteString) 函数。
```python
from itertools import chain # 【可在代码顶部导入】

logger.info("Online ASR extract the feat")
samples = np.frombuffer(samples, dtype=np.int16) 
assert samples.ndim == 1     # 是一维数组，单通道的。
#---以上3行为原始代码。便于查看编辑的位置。

middle = []    # 新建一个列表，用来线性插值。
for n,a in enumerate(samples):
    b = samples[n+1] if n<len(samples)-1   else samples[-1]
    middle.append(int((a+b)/2))
samples  = np.array(list (chain.from_iterable(zip(samples,middle))))  # 替换原变量，这样侵入性最小。

# 下接后续代码
```

由于离线识别实际上可以兼容8K语音，自动执行了上采样，就不再单独修改相关的代码了。

## 对API的定制
关于ASR中间过程不返回识别结果，只在接到end信号时返回，需要修改 paddlespeech/server/ws/asr_api.py
主要针对message 包含 bytes字段的情况，注释掉  websocket.send_json() 相关语句即可。只需要保留上面接到end返回最终结果的语句即可。
另外添加暂停、恢复相关信号，讲清楚较为麻烦，直接将验证通过的py文件粘贴到相应目录即可。


## 一些思考
理论上讲，升采样之后需要抗镜像滤波。使用scipy.signal中的resample可以完成。scipy.signal.resample(x,num)  
其中num是int类型，是经过resample之后的符号长度，不是上采样率。不过有时采样比例不是整数时会有偏差，比如期待输出5个符号，可能输出4个，输出是numpy array类型。
但是目前没有进行抗镜像滤波，也没发现什么不好的结果。这一点就先忽略了。


客户端方面，也需要做一些改变。




