

## 文件主要内容：

1. fake_image里面是一些假的ai图像

2. 两个pipeline文件需要搭配从Hugging face下载的现成模型，pipeline应用的是模型推理，需要微调的话可以参考我写的BERT部分。

- 推荐Hugging face上表现好的模型：

- 图像（这个模型是我测试过在Hugging face上最好用的）：[DhruvJariwala/deepfake_vs_real_image_detection · Hugging Face](https://huggingface.co/DhruvJariwala/deepfake_vs_real_image_detection)

- 语音：[DavidCombei (David Combei) (huggingface.co)](https://huggingface.co/DavidCombei)

  

3. 两个Just_test_the_xxxx_model也是结合Hugging face模型的，用于测试模型好不好
4. Json读取是用于OpenForensics数据集的标签文件.json，使用该.py文件会得到文件名和标签的一一对应json文件。这个数据集表现力很强，下面有相关链接。有了数据集和标签之后就可以开始准备训练了。
5. 做深伪识别的话，常遇到视频数据集，这个时候需要对视频做切帧处理以获得大量图像。对于图像的话，首先需要人脸定位，把人脸提取下来再做识别，或者叫extract，我这里提供face_extract.py可供参考。

视频切帧、图像头像获取算法可以参考：[FaceForensics/classification/detect_from_video.py at master · ondyari/FaceForensics · GitHub](https://github.com/ondyari/FaceForensics/blob/master/classification/detect_from_video.py#L175)

# 为什么要做深伪检测

**技术发展背景**

深度伪造技术最早出现在2017年底，由一名Reddit用户发布利用名人面孔合成的色情视频而引发关注。这项技术的背后是深度学习算法，尤其是生成性对抗网络（GAN）和卷积神经网络（CNN），它们使得视频、音频、文本等的伪造变得更加逼真。

在过去的几年里，使用AI的自动视频编辑技术领域取得了巨大进步。尤其是对于面部的处理方法，现在所有人都可以轻松地实现面部重现（即将面部表情从一个视频转移到另一个视频）。这使得人们可以毫不费力地改变说话人的身份。

现在面部处理工具已经非常先进，没有任何照片处理和数字处理经验的用户都可以使用它们。并且随着时间的推进，有越来越多免费的代码和库向用户开放。

**为什么要做深伪检测？**

深伪换脸技术的进步不仅给新的艺术可能性打开了大门，但不怀好意的用户可以使用深伪技术传播虚假新闻和虚假的色情技术。这类视频或者媒体的传播会导致许多严重的后果，比如网络诈骗、侵犯个人权益、政治舆论操纵、色情内容的非法制作和传播以及公众对媒体内容的信任度下降等。

如：

1. **乌克兰总统虚假视频**：一条伪造的乌克兰总统泽连斯基的视频被广泛传播，视频中伪造的泽连斯基呼吁乌克兰士兵放下武器投降，乌克兰国防部随后进行了辟谣。
2. **美国政治人物虚假电话**：美国新罕布什尔州的一些选民接到了伪造的自动留言电话，声称是“拜登总统”告诉他们不要在该州初选中投票，后经美国白宫新闻秘书确认电话内容为伪造。
3. **肖像权侵权案件**：古风汉服网红魏某起诉了4家运营AI换脸软件的公司，因为这些公司在未经授权的情况下使用了她的肖像生成换脸视频，侵犯了其肖像权。
4. **香港AI诈骗案**：诈骗者利用深度伪造技术仿造了一家英国公司高层管理人员的形象和声音，通过视频会议冒充公司高层，骗取了香港分公司财务职员2亿港元。

根据McAfee公布的最新报告，基于人工智能(AI) 的语音诈骗日益猖獗，在接到诈骗电话的群体中，**77%**的人会导致经济损失。结果显示，约**31%**的人差点被骗，**18%**则已经被骗。 

我们希望为用户带来深伪技术的提醒，在用户接到电话或者在视频中能为用户识别对方是否是真人，并且提示用户以减少经济损失。



**深伪检测的价值：**

由于深伪内容的逼真性，普通用户难以使用肉眼区分真伪，这就需要有效的检测技术来鉴别伪造内容，减少其对社会和个人的负面影响。



深伪语音数据集：

| ID   | 数据集名称                                     | 数据集介绍                                                   | 是否需要申请               | 数据集链接                                                   |
| ---- | ---------------------------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| 1    | Fake-or-Real (FoR)                             | 是一个用于训练分类器以检测合成语音的大型集合，它包含了超过195,000条由真实人类和计算机生成的语音话语。这个数据集整合了最新的文本到语音(Text-to-Speech, TTS)技术解决方案，如 Deep Voice 3 和 Google Wavenet TTS，以及多样的真实人类语音样本，包括 Arctic 数据集、LJSpeech 数据集、VoxForge 数据集以及一些自主收集的语音录音。 | 是                         | https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset |
| 2    | H-Voice                                        | 基于模仿方法生成的假音频数据集，包含了6672个样本，并以直方图的形式保存为**PNG格式** | 否，以图片保存的，不够通用 | [H-Voice: Fake voice histograms (Imitation+DeepVoice) - Mendeley Data](https://data.mendeley.com/datasets/k47yd3m28w/4) |
| 3    | ASV Spoof Challenge 2019                       | 用于自动说话人验证**欺骗**和对策挑战的数据集，包含了**合成和模仿**的音频样本 | 是                         | https://www.asvspoof.org/index2019.html                      |
| 4    | ASV Spoof 2021 Challenge                       | 包含逻辑和物理两种假音频场景，逻辑场景使用合成软件制作假音频，物理场景是通过使用真实说话者数据的部分来复制预先录制的音频。 | 是                         |                                                              |
| 5    | Ar-DAD Arabic Diversified Audio                | 包含来自《古兰经》音频门户的阿拉伯语说话者原始和模仿的声音，涵盖了**沙特阿拉伯、科威特、埃及、也门、苏丹和阿联酋的阿拉伯人** | 否，但是语言不适合         | [Ar-DAD: Arabic Diversified Audio Dataset - Mendeley Data](https://data.mendeley.com/datasets/3kndp5vs6b/3) |
| 6    | M-AILABS Speech                                | 一个**德语**音频数据集，用于语音识别和合成音频，包含9265个真实音频样本和806个假样本 | 资源已经关闭               | 资源已经关闭 https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/ |
| 7    | FakeAVCeleb                                    | 一个限制性的英语说话者（**明星**）数据集，由SV2TTS工具合成生成，包含490个真实样本和20000个假样本 | 是                         | [FakeAVCeleb (google.com)](https://sites.google.com/view/fakeavcelebdash-lab/) |
| 8    | Audio Deep synthesis Detection challenge (ADD) | 包含低质量假音频检测（LF）、部分假音频检测（PF）和假音频游戏（FG），其中LF包含带有真实世界噪音的300个真实声音和700个完全假的说话词。 | 否，属于场景ai，不是人声ai | [ADD 2023 (addchallenge.cn)](http://addchallenge.cn/databases2023) |
| 9    | FakeSound                                      | FakeSound数据集是基于自动化操纵管道生成的，该管道利用了高性能的定位、再生和超分辨率模型来有效生成深度伪造的通用音频。 | 是                         | https://fakesounddata.github.io/                             |

深伪图像/视频数据集：

|  ID  |   数据集名称    |                          数据集内容                          | MOS意见分数（逼真程度） |                          数据集链接                          |
| :--: | :-------------: | :----------------------------------------------------------: | :---------------------: | :----------------------------------------------------------: |
|  1   |  OpenForensics  | 数据集构建工作流主要包括真人图像采集、伪造人脸图像合成和多任务标注三个步骤。160,670张真实人脸173,660张伪造人脸 |           4.0           |           https://github.com/ltnghia/openforensics           |
|  2   |    Celeb-DF     | 包含不同年龄、种族和性别的名人视频和对应的伪造视频590真实名人视频5639虚假名人视频 |           3.2           |     https://github.com/yuezunli/celeb-deepfakeforensics      |
|  3   | DeeperForensics |     60000视频，共1760万帧图像。54000真实视频6000伪造视频     |           2.8           | 申请制：[DeeperForensics-1.0/dataset at master · EndlessSora/DeeperForensics-1.0 (github.com)](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset#download) |
|  4   | FaceForensic++  |             1000真实视频4000伪造视频1800万张图像             |           1.3           | [FaceForensics/dataset at master · ondyari/FaceForensics (github.com)](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
