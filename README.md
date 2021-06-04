# Music Source Separation incorporating Generative Adversarial Networks


## モチベーション：音源分離システムの汎化性能の向上
<p align="center">
![image](https://user-images.githubusercontent.com/67317828/120760595-3e1b5900-c54f-11eb-9053-bdd84bb51a0d.png)
</p>
一般に教師あり学習を続けていくと過学習に陥り、訓練誤差と汎化誤差（未知のデータを判定したときの誤差）の間にギャップが生じる。その差異を埋めることで、既存の音源分離システムの性能向上を図る。
## 提案手法：損失関数に自然さの尺度を追加
教師あり学習では、損失関数でシステムの出力の善し悪しを計り、改善されるように学習する。そのため損失関数はシステムの目標といえる。既存手法は、損失関数は正解波形との差分、すなわち信号同士の比較のみで損失関数が定義されているが、提案手法ではこれに加えて自然さを算出する。
### ①Why?
<p align="center">
![image](https://user-images.githubusercontent.com/67317828/120760672-555a4680-c54f-11eb-883b-963723bb473a.png)
</p>
人間は初めて耳にした楽曲でも、特定の楽器音のみを聞き取ることができる。人間が未知のデータに対しても音源分離可能であるのは、音の三要素である音量・音高・音色、またその組み合わせから感じ取れる無数の要素を経験の中で培い、その楽器音としての自然な音をイメージすることができるからだと考える。自然さを構成する無数の要素を学習することを機械で実現するためには、システムの目標となる損失関数が最適であると考えた。
### ②How-to?
<p align="center">
![image](https://user-images.githubusercontent.com/67317828/120760716-673be980-c54f-11eb-8f7f-d81d7c05c20d.png)
</p>
自然な画像を生成できる技術としてGANがある。Generator/Discriminatorと呼ばれる2つのネットワークを競合させる学習方法は、しばしば紙幣の偽造に例えられる。偽造者（Generator）は本物に近い偽札を作ろうとし、警官（Discriminator）はそれが偽物であると見抜く。するとGeneratorは、より精巧な偽札を作り出すように技術を発展させる。こうした「いたちごっこ」が繰り返され、最終的には本物に近い偽札が生成されるようになる。ここでDiscriminatorは、自然さを算出する損失関数のようにふるまう。このアルゴリズムを取り入れることで実現を目指す。
## 検証に用いた音源分離システムと実装環境
世界的なコンペティションSiSec2018において良好な性能を示した、Facebookの音源分離システムを検証に用いた。Encoder-Decoderモデルを採用しており、Encoderで復元可能な情報量を保ったまま特徴抽出を行い、Decoderで音声に復元する構造を持たせている。より生成モデルに近い構造を持ち、混合音から分離音への音声変換のような構造を目指してきたことが伺える。大規模モデルを動作させるために、GCP上で最新のGPUであるA100を16基分散学習させた。使用言語はPython。フレームワークはPyTorch。
## 研究進捗
コンペティションSiSec2018で使用された評価指標SDRによる結果は表の通り。

| Model | Drum | Bass | Others | Vocal | All |
| ------------- |------:|------:|------:|------:|------:|
| [Demucs][demucs] | 6.08 | 5.83 | **4.12** | **6.29** | **5.58** |
| Demucs+GAN | **6.15** | **6.16** | 3.95 | 5.99 | 5.56 |
| Difference | +0.07 | +0.33 | -0.17 | -0.30 | -0.02 |

Discriminatorの過学習が起きている状態であったが、DrumとBassは性能が上がり、OthersとVocalは下がるという結果になった。Discriminatorの過学習を改善したのち、2021年度に国内学会、2022年度に国際会議での発表を目指している。


[nsynth]: https://magenta.tensorflow.org/datasets/nsynth
[sing_nips]: https://research.fb.com/publications/sing-symbol-to-instrument-neural-generator
[sing]: https://github.com/facebookresearch/SING
[waveunet]: https://github.com/f90/Wave-U-Net
[musdb]: https://sigsep.github.io/datasets/musdb.html
[museval]: https://github.com/sigsep/sigsep-mus-eval/
[openunmix]: https://github.com/sigsep/open-unmix-pytorch
[mmdenselstm]: https://arxiv.org/abs/1805.02410
[demucs_arxiv]: https://hal.archives-ouvertes.fr/hal-02379796/document
[musevalpth]: museval_torch.py
[tasnet]: https://github.com/kaituoxu/Conv-TasNet
[audio]: https://ai.honu.io/papers/demucs/index.html
[spleeter]: https://github.com/deezer/spleeter
[soundcloud]: https://soundcloud.com/voyageri/sets/source-separation-in-the-waveform-domain
[original_demucs]: https://github.com/facebookresearch/demucs/tree/dcee007a350467abc3295dfe267034460f9ffa4e
[diffq]: https://github.com/facebookresearch/diffq
[d3net]: https://arxiv.org/abs/2010.01733
[demucs]: https://github.com/facebookresearch/demucs