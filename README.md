# 研究で記述しているコードの紹介

## プログラム

### AVX/AVX2でベクトル化された方向性Cubic補間のプログラム（C++） - [リンク](https://github.com/ngmhrk/sampleCodesForAcademics/blob/main/sampleCodesForAcademics/DCCI32FC1.cpp#L6)

- 方向性Cubic補間（Directional Cubic Convolution Interpolation: DCCI）のプログラム
- 研究室のOBが記述したプログラムを必要に応じて修正（フルスクラッチで書いてはいない）

※プログラムの公開の許可を担当教員から受けています．

### ドメイン固有言語Halideで記述された方向性Cubic補間のプログラム（Halide） - [リンク](https://github.com/ngmhrk/sampleCodesForAcademics/blob/main/sampleCodesForAcademics/DCCI32FC1Halide.cpp#L102)

- 方向性Cubic補間のHalide実装

### Halide

- 高速画像処理に特化したドメイン固有言語（Domain-specific language: DSL）
- C++の内部DSL
- 特徴
    - アルゴリズムと計算スケジュールを分離して記述可能
        - アルゴリズム：どのような処理を行いたいのか（畳み込み演算をするのか，色変換をするのかなどなど）
        - 計算スケジュール：どのように計算するのか（計算順序や，並列化，ベクトル化，ループ構造など）
    - アルゴリズム本体の記述を書き換えることなくスケジュール部分のみの変更で高速化可能

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

以下の環境変数にライブラリのパスを通しています

- OPENCV_INCLUDE_DIR: OpenCVのincludeディレクトリ
- OPENCV_LIB_DIR: OpenCVのlibディレクトリ
- HALIDE_INCLUDE_DIR: Halideのincludeディレクトリ
- HALIDE_LIB_DIR: Halideのlibディレクトリ
- Path: OpenCV/Halideのbinディレクトリ
