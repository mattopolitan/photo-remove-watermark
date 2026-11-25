# 照片去除浮水印 / Photo Watermark Remover

本機端的 AI 浮水印／印章移除工具，使用 RTX 5080 (或任何 CUDA 顯示卡) 在本機執行，無需上傳照片到雲端。

- **偵測**：[YOLO11x watermark detector](https://huggingface.co/corzent/yolo11x_watermark_detection)（MIT 授權）自動找出照片中的浮水印與印章，加上對 `stamps/` 資料夾內印章圖樣的多尺度模板比對。
- **判定**：若照片**沒有**偵測到浮水印或印章，檔案會**原封不動**直接複製到輸出資料夾，不做任何修改。
- **移除**：偵測到的區域以 [LaMa (big-lama)](https://github.com/advimman/lama) 修補模型填補。
- **遞迴處理**：自動掃描資料夾及所有子資料夾內的圖片（jpg / jpeg / png / webp / bmp / tif / tiff）。

---

A local, offline AI tool that removes watermarks and stamps from photos using your RTX 5080 (or any CUDA GPU). Nothing is uploaded to the cloud.

- **Detect**: [YOLO11x watermark detector](https://huggingface.co/corzent/yolo11x_watermark_detection) (MIT license) locates watermarks/stamps automatically, plus multi-scale template matching against stamp images you provide in `stamps/`.
- **Decide**: if nothing is detected, the file is copied to the output folder **completely untouched** — zero modification.
- **Remove**: detected regions are inpainted with [LaMa (big-lama)](https://github.com/advimman/lama).
- **Recursive**: scans a folder and all of its subfolders (jpg / jpeg / png / webp / bmp / tif / tiff).

---

## 使用方法 / Usage

### 1. 複製此 repo / Clone this repo

```bash
git clone <your-repo-url> photo-remove-watermark
cd photo-remove-watermark
```

### 2. 把照片放進 `photos/` 資料夾 / Put your photos into `photos/`

可以包含任意層级的子資料夾，程式會遞迴掃描。原圖不會被修改，結果輸出到 `photos_cleaned/`（保持相同的目錄結構）。

Subfolders are fine — everything is scanned recursively. Originals are never modified; results go to `photos_cleaned/` with the same folder structure.

### 3. 一鍵執行 / Run with one command

**Linux / macOS:**

```bash
./run.sh
```

**Windows:**

```bat
run.bat
```

首次執行會自動建立 Python 虛擬環境、安裝相依套件，並下載兩個模型權重（偵測模型約 100 MB，LaMa 約 200 MB，皆儲存在本機快取，之後無需重新下載）。之後每次執行只會更新套件並直接開始處理照片。

The first run creates a Python virtual environment, installs dependencies, and downloads the two model weights (detector ~100 MB, LaMa ~200 MB, cached locally — downloaded only once). Later runs go straight to processing.

---

## 進階參數 / Advanced options

直接用 Python 執行可自訂參數（跳過 `run.sh`）：

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python remove_watermark.py --input photos --output photos_cleaned [options]
```

| 參數 / Option | 預設 / Default | 說明 / Description |
|---|---|---|
| `--input` | `photos` | 輸入資料夾（遞迴掃描）/ Input folder (scanned recursively) |
| `--output` | `photos_cleaned` | 輸出資料夾 / Output folder |
| `--conf` | `0.35` | 偵測信心值門檻（0–1）。漏偵測就調低、誤判就調高 / Detection threshold. Lower = catch more, higher = fewer false positives |
| `--padding` | `8` | 偵測框外擴像素數，確保浮水印邊緣完整覆蓋 / Extra pixels around each detected box |
| `--dilate` | `12` | 遮罩膨脹大小，修補更自然 / Mask dilation for smoother inpainting |
| `--device` | 自動 / auto | `cuda` 或 `cpu` / Force `cuda` or `cpu` |
| `--stamps` | `stamps` | 印章模板資料夾；每張模板會在多個縮放比例下與每張照片比對，命中即修補。資料夾不存在或為空時自動跳過 / Stamp-template folder. Each template is matched against every photo at multiple scales; matches are inpainted. Skipped if missing or empty |
| `--stamp-threshold` | `0.72` | 模板比對相似度門檻（0–1）。漏掉就調低、誤判就調高 / Template-match threshold. Lower = catch more, higher = fewer false positives |

---

## 用印章模板移除偵測不到的印章 / Removing missed stamps with templates

YOLO 偵測模型認不出的印章（例如特殊造型的徽章貼紙），可以用**模板比對**移除：

1. **裁切印章圖樣** —— 用 `extract_stamps.py` 從任何一張有該印章的照片裁出來（先不加 `--box` 可以看到圖片尺寸）：

   ```bash
   python extract_stamps.py photos/84198914.jpeg                          # 只看尺寸
   python extract_stamps.py photos/84198914.jpeg --box 6,52,146,228 --out stamps/badge.png
   ```

   `--box` 是 `x1,y1,x2,y2` 像素座標（左上角到右下角）；也可用 `--rel` 改成 0–1 比例。裁剪範圍緊貼印章即可，程式會自動嘗試多種縮放比例。

2. **放進 `stamps/` 資料夾** —— 可以同時放多張模板，每張都會被比對。

3. **重新執行 `run.bat`** —— 所有命中模板的區域都會和 YOLO 偵測結果一起交給 LaMa 修補。不想再用某張模板時，把它從 `stamps/` 刪掉即可。

If the YOLO detector misses a stamp (unusual badge stickers, etc.), remove it with **template matching**:

1. **Crop the stamp** from any photo that contains it (run without `--box` to see the image size):

   ```bash
   python extract_stamps.py photos/84198914.jpeg                          # size only
   python extract_stamps.py photos/84198914.jpeg --box 6,52,146,228 --out stamps/badge.png
   ```

   `--box` is `x1,y1,x2,y2` in pixels (top-left to bottom-right); add `--rel` for 0–1 fractions. Crop tight around the stamp — multiple scales are tried automatically.

2. **Drop it into `stamps/`** — several templates can live there at once.

3. **Re-run `run.bat`** — matched regions are inpainted together with YOLO detections. Delete a template from `stamps/` to stop matching it.

---

## 環境需求 / Requirements

- NVIDIA 顯示卡 + 最新 [NVIDIA Driver](https://www.nvidia.com/download/index.aspx)（RTX 5080 需 570 以上版本）
- Python 3.10 – 3.13（`run.bat` 偵測到 [uv](https://docs.astral.sh/uv/) 時會自動改用 standalone Python 3.13 建立環境；Linux/macOS 直接使用系統 `python3`）
- 約 8 GB 磁碟空間（套件 + 模型權重）
- `requirements.txt` 已設定從 PyTorch 官方 `cu128` 通道安裝，以支援 RTX 50 系列（Blackwell）顯示卡。

An NVIDIA GPU with up-to-date drivers (RTX 5080 needs driver ≥ 570), Python 3.10–3.13, and ~8 GB disk space. On Windows, `run.bat` bootstraps a standalone Python 3.13 via [uv](https://docs.astral.sh/uv/) when available; on Linux/macOS the system `python3` is used. `requirements.txt` installs PyTorch from the official `cu128` index, which supports RTX 50-series (Blackwell) cards.

執行時若出現 `CUDA out of memory`，可改用較小的偵測模型（見下）。

If you hit `CUDA out of memory`, you can swap in a smaller detector model (see below).

---

## 常見問題 / Troubleshooting

**偵測不到浮水印 / Watermark not detected**
調低 `--conf`（例如 `--conf 0.2`）。透明、極淡的全圖平鋪浮水印對偵測模型較困難，效果因圖而異。

Lower `--conf` (e.g. `--conf 0.2`). Faint, tiled full-image watermarks are inherently hard to detect; results vary by image.

**印章沒被移除 / A stamp was not removed**
偵測模型對特殊造型或半透明的印章可能視而不見，這時改用[模板比對](#用印章模板移除偵測不到的印章--removing-missed-stamps-with-templates)：裁一張模板放進 `stamps/` 即可，無需調整其他設定。

The detector can miss unusual or semi-transparent stamps. Use [template matching](#用印章模板移除偵測不到的印章--removing-missed-stamps-with-templates) instead: crop one template into `stamps/` and re-run — no other changes needed.

**誤判（乾淨照片被修改）/ False positives**
調高 `--conf`（例如 `--conf 0.5`）。記住：被誤判的照片是輸出「修改版」，原始檔案永遠不動，可直接從 `photos/` 複製回來。

Raise `--conf` (e.g. `--conf 0.5`). False-positive edits only affect the *output* copy — originals in `photos/` are never touched.

**`torch.cuda.is_available()` 回傳 False / CUDA not available**
確認已安裝 NVIDIA 驅動，並檢查安裝 log 是否出現 `+cu128` 字樣：

```bash
pip uninstall -y torch torchvision
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**想換其他偵測模型 / Swapping the detector**
修改 `remove_watermark.py` 開頭的 `DETECTOR_REPO` / `DETECTOR_FILE`，任何 Ultralytics YOLO 格式的 `.pt` 權重皆可使用（例如較小的 `corzent/yolov12x-dino3-watermark_detection` 或 [Roboflow 上的印章模型](https://universe.roboflow.com/)）。

---

## 授權與免責聲明 / License & Disclaimer

偵測模型為 [MIT](https://huggingface.co/corzent/yolo11x_watermark_detection)；LaMa 為 [Apache-2.0](https://github.com/advimman/lama)。本專案程式碼以 MIT 授權釋出。

請只移動**你自己擁有版權**的照片上的浮水印與印章。移除他人照片的版權浮水印可能違反著作權法。

Detector: [MIT](https://huggingface.co/corzent/yolo11x_watermark_detection); LaMa: [Apache-2.0](https://github.com/advimman/lama). Code in this repo is MIT.

Only remove watermarks/stamps from photos **you own the rights to**. Removing copyright marks from others' photos may violate copyright law.
