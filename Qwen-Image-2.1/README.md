# Qwen-Image-2.1 Gradio App

Gradio を使用して **Qwen-Image-2.1** による画像生成および画像編集（Image-to-Image / Editing）を行うための Web アプリケーションです。

---

## 🚀 クローン（取得）方法

本リポジトリはフォルダ毎に独立しています。`Qwen-Image-2.1` フォルダのみを取得して利用してください。

### 1. Node.js がインストールされている場合（推奨）

`npx degit` を利用することで、`Qwen-Image-2.1` フォルダのみを高速に取得できます。

```bash
npx degit dai-ichiro/enjoyDiffusers/Qwen-Image-2.1 qwen
cd qwen
```

### 2. Node.js がない場合（Git Sparse Checkout を使用）

Git の Sparse Checkout 機能を使用して、`qwen` フォルダにクローンし、特定フォルダのみをダウンロードします。

```bash
git clone --filter=blob:none --sparse https://github.com/dai-ichiro/enjoyDiffusers.git qwen
cd qwen
git sparse-checkout set Qwen-Image-2.1
cd Qwen-Image-2.1
```

---

## ⚙️ 依存ライブラリのインストール

`uv` を利用してパッケージ管理および実行を行う例：

```bash
uv sync
```

※ Python `>=3.13, <3.14` および CUDA 13.2（PyTorch 2.14.0+cu132）の環境が必要です（本プロジェクトの実行環境は CUDA 13.2 に限定しています）。

> **⚠️ 注意事項（`pyproject.toml` の設定）**
> `pyproject.toml` の最終行にある `TORCH_CUDA_ARCH_LIST`（デフォルト表記例: `8.6`）は、ビルド時に使用する GPU の Compute Capability（アーキテクチャ）に合わせて各自変更が必要です。（例: RTX 3090/3080等なら `8.6`、RTX 4090/4080等なら `8.9` など）

---

## 📦 モデルのダウンロード

アプリの実行に必要なモデルファイルを `./models/` フォルダ内に配置します。
付属の `download_and_quantization.py` や Hugging Face CLI / `huggingface_hub` を使用してダウンロードしてください。

### 必要とされるモデルディレクトリ構成

```text
models/
├── Qwen-Image-2.1/                         # ベースモデル
├── Qwen-Image-2.1_bnb_4bit/                # bnb-4bit 量子化モデル（任意）
├── Qwen_Image_2_1_sdnq_dynamic_4bit/       # SDNQ 4bit 量子化モデル（任意）
├── Qwen_Image_2_1_sdnq_dynamic_8bit/       # SDNQ 8bit 量子化モデル（任意）
├── Qwen-Image-2.1-PE-T2I/                  # プロンプト拡張 (Text-to-Image) 用モデル
└── Qwen-Image-2.1-PE-I2I/                  # プロンプト拡張 (Image-to-Image) 用モデル
```

`download_and_quantization.py` を編集・実行して必要なモデルを取得できます：

```bash
python download_and_quantization.py
```

---

## 🎈 アプリの起動

モデルの準備ができたら、以下のコマンドで Gradio アプリを起動します。

```bash
python app.py
```

`uv` をお使いの場合:

```bash
uv run app.py
```

起動後、ターミナルに表示される URL（通常は `http://127.0.0.1:7860`）へブラウザでアクセスしてください。

---

## ✨ 主な機能と使い方

1. **Text-to-Image (T2I) & Image Editing (I2I)**
   - 画像をアップロードしない場合は **Text-to-Image** モードになります。
   - 1〜10 枚の参照画像をアップロードすると自動的に **Image Editing** モードになり、アップロード順に画像を参照して編集指示を出すことができます。

2. **プロンプト拡張 (Enhance Prompt / Prompt Extend)**
   - チェックを有効にすると、専用の LLM モデル (PE-T2I / PE-I2I) がプロンプトを自動で詳細化・最適化します。
   - 画像サイズ（アスペクト比）も拡張モデルの出力結果に基づいて自動選択されます。

3. **量子化 (Quantization) 選択**
   - 詳細設定 (Advanced Settings) から `normal (no quantization)`、`bnb-4bit`、`sdnq-4bit`、`sdnq-8bit` などの量子化バリアントを切替可能です。

4. **透過画像 (RGBA) 生成**
   - アルファチャンネルを含む透過画像の生成に対応しています。プロンプトに以下のような指示を含めてください：
     ```text
     This is an RGBA image with transparency. [描きたい内容] The image has alpha channel and the background is transparent.
     ```

5. **VAE tiling / slicing**
   - 大解像度（16:9 や 3:2 など）生成時の VRAM 消費を抑える VAE tiling 機能に対応しています。
