# seminar
テスト
やっほー
伊右衛門

# 12/29にやったこと
.
├── README.md
├── data.py
├── models/
│   ├── generator.py
│   └── discriminator.py
└── data/

・data.pyにデータパイプラインの部分が入っている。
・generatorのコードなどはmodelsディレクトリ内のファイルに書いていく感じがいいかな。
## モデル構造: Generator (生成器)

本プロジェクトの Generator は、**DCGAN (Deep Convolutional GAN)** のアーキテクチャを採用しており、100次元の乱数（ノイズ）を入力として、32x32ピクセルの手書き数字画像を生成している。

### アーキテクチャ概要
Generatorは、**転置畳み込み (Transposed Convolution)** を用いて、特徴マップのサイズを段階的に拡大（アップサンプリング）していく構造になっている。

1.  **入力**: 100次元の潜在変数 $z$ (正規分布ノイズ)
2.  **層の構成**: 全4層のアップサンプリング
    -   入力 $\rightarrow$ $4 \times 4$
    -   $4 \times 4$ $\rightarrow$ $8 \times 8$
    -   $8 \times 8$ $\rightarrow$ $16 \times 16$
    -   $16 \times 16$ $\rightarrow$ $32 \times 32$ (出力画像)
3.  **出力**: $1 \times 32 \times 32$ のグレースケール画像 (値の範囲は $[-1, 1]$)

### 実装のポイント
-   **ConvTranspose2d (逆畳み込み)**:
    通常の畳み込み（画像サイズを小さくする）とは逆に、画像サイズを倍々に拡大しながら特徴を学習している。
-   **Batch Normalization**:
    各層の出力の分布を整えることで、学習を安定させ、勾配消失や初期値依存の問題を軽減している。
-   **活性化関数**:
    -   中間層: **ReLU** (Rectified Linear Unit) を使用し、スパースな表現を学習させている。
    -   出力層: **Tanh** (Hyperbolic Tangent) を使用。これは、Discriminatorへの入力画像が `[-1, 1]` に正規化されているため、出力範囲を合わせるために必須である。

### なぜ 32x32 なのか？
MNISTの元データは $28 \times 28$ ですが、本モデルでは $32 \times 32$ にリサイズして扱っている。
これは、CNNの構造上、サイズを $2$ の累乗（$4 \to 8 \to 16 \to 32$）で変化させる方が、パディング計算が容易でモデル構造がシンプルになるため。
