# FLUX-3-Image-Klein Gradio App

Gradio を使用して **FLUX-3-Image-Klein** による画像生成および画像編集（Image-to-Image / Editing）を行うための Web アプリケーションです。

---

## クローン（取得）方法

本リポジトリはフォルダ毎に独立しています。`FLUX-3-Image-Klein` フォルダのみを取得して利用してください。

### 1. Node.js がインストールされている場合（推奨）

`npx degit` を利用することで、`FLUX-3-Image-Klein` フォルダのみを高速に取得できます。

```bash
npx degit dai-ichiro/enjoyDiffusers/FLUX-3-Image-Klein klein
cd klein
```

### 2. Node.js がない場合（Git Sparse Checkout を使用）

Git の Sparse Checkout 機能を使用して `klein` フォルダにクローンし、`FLUX-3-Image-Klein` フォルダのみをダウンロードします。
クローン後に中身を `klein` フォルダ直下に移動することで、`npx degit` を使用した場合と同じフォルダ構造（`klein/*`）にできます。

```bash
git clone --filter=blob:none --sparse https://github.com/dai-ichiro/enjoyDiffusers.git klein
cd klein
git sparse-checkout set --no-cone FLUX-3-Image-Klein
mv FLUX-3-Image-Klein/* . 2>/dev/null; mv FLUX-3-Image-Klein/.* . 2>/dev/null
rmdir FLUX-3-Image-Klein
```

---

## 環境構築

### 依存ライブラリのインストール

`uv` を利用してパッケージ管理および実行を行う例：

```bash
uv sync
```

※ Python `>=3.13, <3.14` および CUDA 13.2（PyTorch 2.14.0+cu132）の環境が必要です（本プロジェクトの実行環境は CUDA 13.2 に限定しています）。

> **⚠️ 注意事項（`pyproject.toml` の設定）**
> `pyproject.toml` の最終行にある `TORCH_CUDA_ARCH_LIST`（デフォルト表記例: `8.6`）は、ビルド時に使用する GPU の Compute Capability（アーキテクチャ）に合わせて各自変更が必要です。（例: RTX 3090/3080等なら `8.6`、RTX 4090/4080等なら `8.9` など）

---

### prompt upsamplerの量子化

bitsandbytesを使って4bitに量子化します。

```bash
uv run upsampler-quantize.py
```

---

## アプリの起動

モデルの準備ができたら、以下のコマンドで Gradio アプリを起動します。

```bash
uv run app.py
```

---

## ✨ 主な機能と使い方

1. **Text-to-Image (T2I) & Image Editing (I2I)**
   - 画像をアップロードしない場合は **Text-to-Image** モードになります。
   - 1〜10 枚の参照画像をアップロードすると自動的に **Image Editing** モードになり、アップロード順に画像を参照して編集指示を出すことができます。

2. **プロンプト拡張 (Enhance Prompt / Prompt Extend)**
   - チェックを有効にすると、プロンプトを自動で詳細化・最適化します。

3. **量子化 (Quantization) 選択**
   - 詳細設定 (Advanced Settings) から `sdnq-4bit`、`sdnq-8bit` の量子化バリアントを切替可能です。

