# 研究で記述しているコードの紹介

## プログラム

### AVX/AVX2でベクトル化された方向性Cubic補間のプログラム（C++）

- 画像拡大アルゴリズムの1つである，方向性Cubic補間（Directional Cubic Convolution Interpolation: DCCI）のプログラムです．
- 私個人がフルスクラッチで記述したものではなく，研究室のOBが記述したプログラムを必要に応じて修正しているものです．

※私自身は，このSIMDプログラムを0から記述するほどSIMD演算プログラミングに精通してはいません．
ただ，プログラムを読んで編集する程度であれば研究活動で行っています．

※プログラムの公開の許可を担当教員から受けています．

### ドメイン固有言語Halideで記述された方向性Cubic補間のプログラム（Halide）

- 方向性Cubic補間を，ドメイン固有言語（Domain-specific language: DSL）のHalideを用いて記述したプログラムです．
- Halideは，高速に動作する画像処理に特化したドメイン固有言語であり，C++に組み込まれて使用することができる内部DSLです．
- Halideの特徴として，どのような処理を行いたいのか（畳み込み演算をするのか，色変換をするのかなどなど）というアルゴリズム本体を記述する部分と，どのように計算するのか（計算順序や，並列化，ベクトル化，ループ構造など）という計算スケジュールを記述する部分を分離して記述可能であるというものが挙げられます．
- この特徴のおかげで，アルゴリズム本体の記述を書き換えることなく，スケジュール部分のみを書き換えることで，画像処理の高速化を試すことができます．

## 実行方法

1. 下記ライブラリ・ソフトウェアのインストール
2. 下記環境変数にライブラリのディレクトリパスを設定
3. Visual Studio 2022で `sampleCodesForAcademics/sampleCodesForAcademics.sln` を開く
4. `x64/Release` の設定で実行

### 必要なライブラリ・ソフトウェア
- Visual Studio 2022
- [Halide 13.04](https://halide-lang.org/)
- [OpenCV 4.x](https://opencv.org/)

### 環境変数

以下の環境変数にライブラリのPathを通しています

- OPENCV_INCLUDE_DIR: OpenCVのincludeディレクトリ
- OPENCV_LIB_DIR: OpenCVのlibディレクトリ
- HALIDE_INCLUDE_DIR: Halideのincludeディレクトリ
- HALIDE_LIB_DIR: Halideのlibディレクトリ
- Path: OpenCV/Halideのbinディレクトリ