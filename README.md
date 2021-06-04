# Music Source Separation incorporating Generative Adversarial Networks


## モチベーション：音源分離システムの汎化性能の向上

一般に教師あり学習を続けていくと過学習に陥り、訓練誤差と汎化誤差（未知のデータを判定したときの誤差）の間にギャップが生じる。その差異を埋めることで、既存の音源分離システムの性能向上を図る。

## 提案手法：損失関数に自然さの尺度を追加

教師あり学習では、損失関数でシステムの出力の善し悪しを計り、改善されるように学習する。そのため損失関数はシステムの目標といえる。既存手法は、損失関数は正解波形との差分、すなわち信号同士の比較のみで損失関数が定義されているが、提案手法ではこれに加えて自然さを算出する。

<p align="center">
  <img src="https://user-images.githubusercontent.com/67317828/120768290-f00a5380-c556-11eb-9893-11029a0e2503.png" alt=""/></p>


### ①Why?

人間は初めて耳にした楽曲でも、特定の楽器音のみを聞き取ることができる。人間が未知のデータに対しても音源分離可能であるのは、音の三要素である音量・音高・音色、またその組み合わせから感じ取れる無数の要素を経験の中で培い、その楽器音としての自然な音をイメージすることができるからだと考える。自然さを構成する無数の要素を学習することを機械で実現するためには、システムの目標となる損失関数が最適であると考えた。

<p align="center">
  <img src=https://user-images.githubusercontent.com/67317828/120767381-1aa7dc80-c556-11eb-8764-2cc423c04648.png alt=""/></p>


### ②How-to?

自然な画像を生成できる技術としてGANがある。Generator/Discriminatorと呼ばれる2つのネットワークを競合させる学習方法は、しばしば紙幣の偽造に例えられる。偽造者（Generator）は本物に近い偽札を作ろうとし、警官（Discriminator）はそれが偽物であると見抜く。するとGeneratorは、より精巧な偽札を作り出すように技術を発展させる。こうした「いたちごっこ」が繰り返され、最終的には本物に近い偽札が生成されるようになる。ここでDiscriminatorは、自然さを算出する損失関数のようにふるまう。このアルゴリズムを取り入れることで実現を目指す。

<p align="center">
  <img src=https://user-images.githubusercontent.com/67317828/120764510-2d6ce200-c553-11eb-985e-1cf401408d39.jpg alt=""/></p>


## 検証に用いた音源分離システムと実装環境

世界的なコンペティションSiSec2018において良好な性能を示した、Facebookの音源分離システムを検証に用いた。Encoder-Decoderモデルを採用しており、Encoderで復元可能な情報量を保ったまま特徴抽出を行い、Decoderで音声に復元する構造を持たせている。より生成モデルに近い構造を持ち、混合音から分離音への音声変換のような構造を目指してきたことが伺える。大規模モデルを動作させるために、GCP上で最新のGPUであるA100を16基分散学習させた。使用言語はPython。フレームワークはPyTorch。

<p align="center">
  <img src=https://user-images.githubusercontent.com/67317828/120769338-fcdb7700-c557-11eb-9494-f5e6e233baba.png alt=""/></p>


## 研究進捗

コンペティションSiSec2018で使用された評価指標SDRによる結果は表の通り。

| Model | Drum | Bass | Others | Vocal | All |
| ------------- |------:|------:|------:|------:|------:|
| [Demucs][demucs] | 6.08 | 5.83 | **4.12** | **6.29** | **5.58** |
| Demucs+GAN | **6.15** | **6.16** | 3.95 | 5.99 | 5.56 |
| Difference | +0.07 | +0.33 | -0.17 | -0.30 | -0.02 |


Discriminatorの過学習が起きている状態での結果であるため、改善でき次第、更新予定である。


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