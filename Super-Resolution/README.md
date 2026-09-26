# Image Upscaler (Super-Resolution) Gradio App

`image_gen_aux` と Gradio を使用してローカル環境で画像超解像（Upscale）を行うための Web アプリケーションです。

---

## 🚀 クローン（取得）方法

本リポジトリはフォルダ毎に独立しています。`Super-Resolution` フォルダのみを取得して利用してください。

### 1. Node.js がインストールされている場合（推奨）

`npx degit` を利用することで、`Super-Resolution` フォルダのみを高速に取得できます。

```bash
npx degit dai-ichiro/enjoyDiffusers/Super-Resolution super-resolution
cd super-resolution
```

### 2. Node.js がない場合（Git Sparse Checkout を使用）

Git の Sparse Checkout 機能を使用して `super-resolution` フォルダにクローンし、`Super-Resolution` フォルダのみをダウンロードします。
クローン後に中身を `super-resolution` フォルダ直下に移動することで、`npx degit` を使用した場合と同じフォルダ構造にできます。

```bash
git clone --filter=blob:none --sparse https://github.com/dai-ichiro/enjoyDiffusers.git super-resolution
cd super-resolution
git sparse-checkout set --no-cone Super-Resolution
mv Super-Resolution/* . 2>/dev/null; mv Super-Resolution/.* . 2>/dev/null
rmdir Super-Resolution
```

---

## ⚙️ 依存ライブラリのインストール

`uv` を利用してパッケージ管理および実行を行う例：

```bash
uv sync
```

---

## 📦 モデルのダウンロード

アプリの実行に必要なモデルファイルを `./models/` フォルダ内に配置します。
付属の `download_models.py` を実行してダウンロードしてください。

```bash
python download_models.py
```

`uv` をお使いの場合:

```bash
uv run download_models.py
```

### 必要とされるモデルディレクトリ構成

```text
models/
├── UltraSharp/
│   └── 4x-UltraSharp.safetensors
├── DAT/
│   ├── DAT_x2.safetensors
│   ├── DAT_x3.safetensors
│   └── DAT_x4.safetensors
├── RealPLKSR/
│   └── 4xNomosWebPhoto_RealPLKSR.safetensors
├── RealWebPhoto/
│   └── 4xRealWebPhoto_v4_dat2.safetensors
├── 4xRemacri/
│   └── 4x_foolhardy_Remacri.safetensors
└── RealESRGAN/
    ├── general_x4v3.safetensors
    └── anime_x4v3.safetensors
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

起動後、ブラウザで表示される Web UI（Gradio）から画像をアップロードし、使用するモデルを選択して「Upscale」を実行してください。

---

## ✨ 対応モデル

- **UltraSharp (x4)**: `Kim2091/UltraSharp`
- **DAT X2 / X3 / X4**: `OzzyGT/DAT_X2`, `OzzyGT/DAT_X3`, `OzzyGT/DAT_X4`
- **RealPLKSR (x4)**: `OzzyGT/4xNomosWebPhoto_RealPLKSR`
- **DAT-2 RealWebPhoto (x4)**: `Phips/4xRealWebPhoto_v4_dat2`
- **4xRemacri**: `OzzyGT/4xRemacri`
- **RealESRGAN general v3 (x4)**: `OzzyGT/RealESRGAN_general_x4v3`
- **RealESRGAN anime v3 (x4)**: `OzzyGT/RealESRGAN_anime_v3`
