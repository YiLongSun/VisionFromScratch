# VisionFromScratch

以 **Pascal VOC 2007+2012** 練習電腦視覺的兩個核心任務：

| 任務 | 訓練資料 | 評估資料 | 主要指標 |
| --- | --- | --- | --- |
| 物件偵測 | VOC2007 + VOC2012 `Main/trainval`（16,551 張） | VOC2007 `Main/test`（4,952 張） | mAP@0.5 |
| 語意分割 | VOC2012 `Segmentation/train`（1,464 張） | VOC2012 `Segmentation/val`（1,449 張） | mIoU |

設計原則：

- **Notebook 只當啟動器**：訓練邏輯全部在 `src/`，Colab 只負責 clone、安裝、執行。
- **一個 YAML 管全部**：資料、模型、optimizer、scheduler、訓練與存檔設定都在同一個檔案，也能用命令列覆寫。
- **可中斷續訓**：以 step 為單位存到 Google Drive，重新執行就自動接著跑，而且結果與不中斷時逐位元相同（CPU 上有測試驗證）。
- **backbone 可替換**：任何 timm 模型（ResNet、ConvNeXt、ViT、DINO / DINOv2 / DINOv3）都能接到任一種 head。

---

## 專案結構

```
VisionFromScratch/
├── configs/
│   ├── detection.yaml            # 物件偵測預設設定（所有可調參數都有註解）
│   └── segmentation.yaml         # 語意分割預設設定
├── notebooks/colab_launcher.ipynb
├── scripts/
│   └── make_slim_zip.py          # 製作精簡版資料集 zip（只用 Python 內建模組）
├── src/
│   ├── config.py                 # 設定 schema（dataclass）與載入
│   ├── registry.py               # 字串 → 類別的註冊表
│   ├── data/                     # VOC Dataset、transforms、可續跑 sampler、collate
│   ├── models/
│   │   ├── backbone.py           # timm backbone（含外部權重載入）
│   │   ├── necks.py              # FPN、Simple Feature Pyramid（ViTDet）
│   │   ├── detectors.py          # Faster R-CNN、RetinaNet、FCOS
│   │   └── segmentors.py         # DeepLabV3+、Semantic FPN、Linear head
│   ├── engine/
│   │   ├── trainer.py            # 訓練迴圈（AMP、梯度累積、存檔、評估、log）
│   │   ├── checkpoint.py         # 安全寫入與自動找最新 checkpoint
│   │   └── optim.py              # optimizer 與 scheduler
│   ├── metrics/                  # VOC mAP、mIoU
│   ├── tasks/                    # 偵測與分割各自的資料、loss、評估、視覺化
│   ├── train.py                  # 訓練入口
│   └── evaluate.py               # 評估入口
└── tests/                        # 單元測試與端對端測試（用假資料，CPU 可跑）
```

---

## 在 Colab 上使用

### 1. 準備資料集

把 `VOC0712.zip` 上傳到 Google Drive（預設路徑 `MyDrive/datasets/VOC0712.zip`）。zip 內的結構應為：

```
VOC0712/VOC_dataset/VOCdevkit/VOC2007/...
VOC0712/VOC_dataset/VOCdevkit/VOC2012/...
```

**建議先做精簡版**：原始 zip 約 4.7GB，其中 VOC2012 test 約 1.6 萬張圖沒有公開標註，完全用不到。在本機執行（只需要 Python 3.10 以上，不必安裝任何套件）：

```bash
python scripts/make_slim_zip.py VOC0712.zip VOC0712_slim.zip
```

精簡版保留所有標註（含 `SegmentationObject`，之後做實例分割會用到）與清單內的圖片，目錄結構不變。上傳後在 notebook 裡把 `DRIVE_ZIP` 改成新檔名即可。

### 2. 執行 notebook

1. 在 Colab 開啟 `notebooks/colab_launcher.ipynb`（可以直接從 GitHub 開：`檔案 → 開啟筆記本 → GitHub`）
2. 執行階段選 **T4 GPU**
3. 修改第 2 個 cell 的設定：`REPO_URL`（改成你的 GitHub repo，必改）、`CONFIG`（`configs/detection.yaml` 或 `configs/segmentation.yaml`）、`DRIVE_ZIP`（zip 在 Drive 上的路徑）
4. 依序執行全部 cell

notebook 會先把 zip 複製到 `/content` 再解壓（直接從 Drive 解壓非常慢），TensorBoard 直接嵌在 notebook 裡顯示。

### 3. 斷線之後

重新連線，**再執行一次全部 cell** 即可。訓練程式啟動時會到 Drive 上的 checkpoint 目錄找最新且可讀的檔案，自動從那個 step 接著訓練。

---

## 設定檔

`configs/` 只有兩個預設設定檔，**每個可調整的參數都列在檔案裡，並以註解說明意義與可選值**：

| 設定檔 | 預設模型 | 檔尾的「常見組合」 |
| --- | --- | --- |
| `detection.yaml` | FCOS + ResNet-50 + FPN | Faster R-CNN、RetinaNet、DINO ResNet-50、ConvNeXt、DINO ViT + Simple Feature Pyramid |
| `segmentation.yaml` | DeepLabV3+ + ResNet-50（OS16） | Semantic FPN、DINOv2 linear probe、DINOv3 ViT + Semantic FPN |

要做新實驗時，可以複製一份再修改，或直接用命令列覆寫（格式 `a.b.c=value`）：

```bash
python -m src.train --config configs/detection.yaml optim.lr=0.005 data.batch_size=4 train.max_epochs=6
python -m src.train --config configs/detection.yaml experiment.name=det_frcnn_r50 \
    model.head.name=faster_rcnn "model.head.kwargs={box_score_thresh: 0.05}" \
    model.backbone.out_indices=[1,2,3,4] model.neck.extra_levels=maxpool
```

- 欄位定義在 `src/config.py`；打錯欄位名稱或型別不符，啟動時會直接報錯。
- 命令列覆寫 `kwargs` 這類 dict 時是**整個取代**（`model.backbone.kwargs={}` 會清空）；只想改其中一個 key 用 `model.head.kwargs.nms_thresh=0.5`。
- dict 寫法冒號後要有空格：`{box_score_thresh: 0.05}`，並用引號包住整個參數。

### 主要區塊

| 區塊 | 用途 | 重要欄位 |
| --- | --- | --- |
| `experiment` | 名稱、任務、輸出位置 | `task: detection \| semantic_seg`、`output_root`（指向 Drive） |
| `data` | 資料集與 DataLoader | `train_sets`、`val_sets`、`batch_size`、`val_max_samples` |
| `data.augment` | 資料增強 | 偵測：`min_size`（多尺度）、`test_min_size`；分割：`scale_range`、`crop_size` |
| `model.backbone` | timm backbone | `name`、`pretrained`、`checkpoint`、`out_indices`、`freeze`、`freeze_bn`、`kwargs` |
| `model.neck` | 特徵整理 | `name: fpn \| simple_pyramid \| none`、`extra_levels: none \| maxpool \| p6p7` |
| `model.head` | 任務 head | 偵測 `faster_rcnn \| retinanet \| fcos`；分割 `deeplabv3plus \| fpn_seg \| linear` |
| `optim` | optimizer | `name: sgd \| adam \| adamw`、`lr`、`weight_decay`、`backbone_lr_mult` |
| `scheduler` | 學習率排程（以 step 計） | `name: constant \| cosine \| multistep \| step \| poly`、`warmup_steps` |
| `train` | 訓練迴圈 | `max_steps` 或 `max_epochs`、`grad_accum_steps`、`grad_clip_norm`、`precision` |
| `checkpoint` | 存檔與續訓 | `save_every_steps`、`keep_last`、`resume: auto \| none \| <路徑>`、`monitor` |
| `logging` | TensorBoard 視覺化 | `num_vis_samples` |

說明：

- `scheduler.milestones` 與 `step_size` 以**總 step 的比例**表示（例如 `[0.67, 0.89]`），改變訓練長度時不必重算。
- `optim.backbone_lr_mult`：backbone 學習率的倍率。微調預訓練 ViT 時常設 0.1。
- `optim.no_weight_decay_norm_bias`：norm 層、bias、ViT 位置編碼不做 weight decay。
- `train.precision`：T4 用 `fp16`（T4 不支援 bf16，設了也會自動改成 fp16）。

---

## 模型組合

模型 = **backbone + neck + head**，三者由 YAML 自由組合，anchor 尺寸等參數依特徵的 stride 自動計算。

### Backbone（timm）

`model.backbone.name` 可以是任何 timm 模型名稱，包含權重 tag。常用選擇：

| 類型 | 名稱範例 | 說明 |
| --- | --- | --- |
| ResNet | `resnet50.a1_in1k`、`resnet50.tv2_in1k` | 偵測／分割最常見的基準 |
| ConvNeXt | `convnext_tiny.fb_in22k_ft_in1k` | 較新的 CNN，通常比 ResNet-50 好 |
| 輕量 | `mobilenetv3_large_100.ra_in1k`、`efficientnet_b3.ra2_in1k` | 速度優先 |
| ViT（監督式） | `vit_small_patch16_224.augreg_in21k_ft_in1k` | |
| DINO ViT | `vit_small_patch16_224.dino`、`vit_base_patch16_224.dino` | 自監督，特徵品質好 |
| DINOv2 ViT | `vit_small_patch14_dinov2.lvd142m`、`vit_base_patch14_dinov2.lvd142m` | patch 14 |
| DINOv3 ViT | `vit_small_patch16_dinov3.lvd1689m`、`vit_base_patch16_dinov3.lvd1689m` | 需 timm ≥ 1.0.20 |
| DINOv3 ConvNeXt | `convnext_tiny.dinov3_lvd1689m` | 自監督 CNN，可直接接 FPN |
| DINO ResNet-50 | 見 `configs/detection.yaml` 檔尾的常見組合 | timm 沒有收錄，用 `checkpoint` 從官方網址載入 |

查詢 timm 支援的權重：

```python
import timm
timm.list_pretrained("*dino*")
```

> DINOv3 權重採用 Meta 的 DINOv3 License。如果 Hugging Face 要求先同意授權才能下載，請到模型頁面同意後，在 Colab 設定 `HF_TOKEN`。

**外部權重**：`model.backbone.checkpoint` 可填本機路徑或 URL，會自動去掉 `module.`、`backbone.` 等前綴並略過用不到的參數（例如分類頭）。若權重放在 checkpoint 的某個 key 底下（例如 DINO 完整 checkpoint 的 `teacher`），設定 `checkpoint_key`。

### Neck

| 名稱 | 適用 | 說明 |
| --- | --- | --- |
| `fpn` | CNN（多尺度特徵） | 自頂向下融合，各層通道數相同 |
| `simple_pyramid` | ViT（單一尺度特徵） | ViTDet 作法：用反卷積／池化從最後一層產生多尺度，`scales` 決定層數 |
| `none` | 分割 head | 直接把 backbone 特徵傳給 head |

`extra_levels`：`maxpool` 加一層 P6 給 Faster R-CNN 的 RPN；`p6p7` 加 P6、P7 給 RetinaNet / FCOS。

### Head

| 任務 | 名稱 | 搭配 |
| --- | --- | --- |
| 偵測 | `faster_rcnn` | backbone `out_indices: [1,2,3,4]`、neck `fpn` + `maxpool` |
| 偵測 | `retinanet`、`fcos` | backbone `out_indices: [2,3,4]`、neck `fpn` + `p6p7` |
| 分割 | `deeplabv3plus` | neck `none`，backbone 輸出 stride 4 與最後一層；CNN 建議 `output_stride: 16` |
| 分割 | `fpn_seg` | neck `fpn` 或 `simple_pyramid` |
| 分割 | `linear` | 凍結 backbone 的 linear probe，適合比較自監督特徵 |

偵測 head 使用 torchvision 的實作，`model.head.kwargs` 會直接傳給 torchvision（例如 `score_thresh`、`nms_thresh`、`detections_per_img`）。

---

## 續訓機制

Checkpoint 存在 `<output_root>/<experiment.name>/checkpoints/step_XXXXXXXX.pt`，內容：

- model、optimizer、scheduler、GradScaler 的 state_dict
- 訓練進度：step、epoch、本 epoch 已訓練的樣本數、目前最佳指標
- RNG 狀態：Python、NumPy、PyTorch CPU、CUDA
- 當時的完整設定

**為什麼能精確接上？**

1. **可續跑的 sampler**：每個 epoch 的資料順序只由 `(seed, epoch)` 決定，續訓時重建同樣的順序，並跳過已經訓練過的樣本。
2. **每個樣本有自己的亂數種子**：資料增強的隨機性與 worker 數量、是否中斷都無關。
3. **DataLoader 使用獨立的 generator**：建立 iterator 時不會動到全域 RNG，模型的 dropout 等隨機性也能一致。
4. **TensorBoard 不重疊**：以 `purge_step` 清掉 checkpoint 之後、斷線之前寫下的紀錄。

**寫入安全**：先存到本機暫存目錄，再複製成 Drive 上的 `*.tmp`，最後改名成正式檔名。斷線時最多留下 `.tmp`，不會出現損壞的正式檔；讀取時若最新檔損壞，會自動退回前一個。

**手動中斷**（按 Colab 的停止鍵）也會先存一次 checkpoint 再結束。

其他檔案：

- `checkpoints/best.pt`：依 `checkpoint.monitor` 的最佳模型（只含權重與指標）
- `config.yaml`：最近一次執行的設定快照
- `final_metrics.json`：訓練結束時在完整驗證集上的指標

> GPU 上 cuDNN 的部分運算本身有非確定性，所以 GPU 上的續訓結果不保證「逐位元」相同，但資料順序、增強與學習率排程都會完全接上。

---

## TensorBoard

log 寫在 `<output_root>/<experiment.name>/tensorboard/`：

| 標籤 | 內容 |
| --- | --- |
| `train/*` | 各項 loss、梯度範數、AMP scale |
| `lr/*` | 每組參數（backbone / head、有無 weight decay）的學習率 |
| `val_partial/*` | 訓練中定期評估（只跑前 `val_max_samples` 張） |
| `val/*` | 訓練結束時的完整評估 |
| `val_vis/*` | 預測視覺化：偵測畫框；分割為「原圖 \| 標註 \| 預測」 |
| `perf/*` | 每秒 step 數、GPU 記憶體峰值 |

偵測指標：`mAP50`（VOC 所有點插值）、`mAP50_voc07`（VOC2007 的 11 點插值）、`mAP50_95`、每類 `AP50/<類別>`。
分割指標：`mIoU`、`pixel_acc`、`mAcc`、每類 `IoU/<類別>`。

---

## 資料集注意事項

- **只讀 ImageSets 清單**：JPEGImages 包含所有任務與切分的圖，不可直接全部讀取。
- **分割不可用 VOC2007 test 評估**：VOC2012 分割 trainval 含有全部 632 張 VOC2007 分割圖（包含 210 張 VOC2007 test），會造成資料洩漏。本專案的分割設定一律使用 VOC2012 val。
- **bbox 座標**：XML 是從 1 開始的像素編號，程式轉成連續座標（左上減 1）。
- **difficult 物件**：訓練時預設排除（`data.use_difficult: false`）；評估時保留，並依 VOC 規則忽略（既不算 TP 也不算 FP）。
- **分割資料量小**：只有 1,464 張訓練圖，mIoU 會比論文低（論文多半用 SBD 擴充到 10,582 張）。

---

## 本機開發

```bash
# 先依 https://pytorch.org 安裝 torch 與 torchvision
pip install -r requirements-dev.txt
pytest                 # 用假資料在 CPU 上跑完所有測試（約 1–2 分鐘）
```

測試涵蓋：VOC mAP 與 mIoU 計算、sampler 續跑、scheduler、外部權重載入、checkpoint 損壞時的退回機制、每種模型組合的訓練與評估，以及「中斷後續訓」與「不中斷」的最終權重逐位元相同。

notebook 直接編輯 `notebooks/colab_launcher.ipynb` 即可。提交到 git 前建議先清除輸出（Colab：`編輯 → 清除所有輸出`），避免把執行紀錄與圖片存進 repo。

### 新增模型

- 新的 neck／分割 head：在對應檔案寫好類別，加上 `@NECKS.register("名稱")` 或 `@SEG_HEADS.register("名稱")`。
- 新的偵測架構：寫一個 `build_xxx(features, num_classes, transform_kwargs, **kwargs)`，加上 `@DETECTORS.register("名稱")`。
- 新的任務：繼承 `src/tasks/base.py` 的 `Task`，加上 `@TASKS.register("名稱")`。

---

## 後續規劃

- [ ] 實例分割：新增 `instance_seg` task，讀取 `SegmentationObject`，head 使用 torchvision `MaskRCNN`（可直接共用目前的 backbone + FPN）
- [ ] 加入 SBD 擴充分割訓練資料
- [ ] 推論腳本：對任意圖片輸出視覺化結果
