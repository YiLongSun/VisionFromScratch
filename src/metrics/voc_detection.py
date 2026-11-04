"""VOC 物件偵測評估：mAP@0.5（官方指標）與 mAP@[0.5:0.95]。

規則與 VOC devkit 相同：
- 每個預測框依分數由高到低，和同一張圖、同一類別中 IoU 最大的 GT 比對
- IoU ≥ 門檻且該 GT 尚未被配對 → TP；GT 已被配對 → FP（重複偵測）；IoU 不足 → FP
- 配到 difficult GT 的預測既不算 TP 也不算 FP；difficult GT 也不計入 GT 總數

由於配對只和「同一張圖內」的預測順序有關，可以逐張圖計算 TP/FP，不必把所有預測留到最後。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from torchvision.ops import box_iou

IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


class VOCDetectionEvaluator:
    def __init__(self, class_names: Sequence[str], iou_thresholds: Sequence[float] = IOU_THRESHOLDS) -> None:
        """class_names 依類別編號 1..N 排列（不含背景）。"""
        self.class_names = list(class_names)
        self.iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._scores: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._tp: Dict[int, List[np.ndarray]] = defaultdict(list)  # 每個元素 shape (T, D)
        self._ignored: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._num_gt = np.zeros(len(self.class_names) + 1, dtype=np.int64)

    @torch.no_grad()
    def update(
        self, predictions: Sequence[Dict[str, torch.Tensor]], targets: Sequence[Dict[str, torch.Tensor]]
    ) -> None:
        """predictions：boxes / scores / labels；targets：boxes / labels / difficult（皆為原圖座標）。"""
        for pred, gt in zip(predictions, targets):
            pred_boxes = pred["boxes"].detach().float().cpu()
            pred_scores = pred["scores"].detach().float().cpu()
            pred_labels = pred["labels"].detach().cpu()
            gt_boxes = gt["boxes"].as_subclass(torch.Tensor).float().cpu()
            gt_labels = gt["labels"].cpu()
            gt_difficult = gt["difficult"].cpu().bool()

            for label in torch.unique(torch.cat([pred_labels, gt_labels])).tolist():
                g_mask = gt_labels == label
                self._num_gt[label] += int((~gt_difficult[g_mask]).sum())
                p_mask = pred_labels == label
                if not p_mask.any():
                    continue
                order = torch.argsort(pred_scores[p_mask], descending=True)
                scores = pred_scores[p_mask][order].numpy()
                tp, ignored = self._match(pred_boxes[p_mask][order], gt_boxes[g_mask], gt_difficult[g_mask].numpy())
                self._scores[label].append(scores)
                self._tp[label].append(tp)
                self._ignored[label].append(ignored)

    def _match(self, det_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_difficult: np.ndarray):
        num_t, num_d = len(self.iou_thresholds), len(det_boxes)
        tp = np.zeros((num_t, num_d), dtype=bool)
        ignored = np.zeros((num_t, num_d), dtype=bool)
        if len(gt_boxes) == 0:
            return tp, ignored
        ious = box_iou(det_boxes, gt_boxes).numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious.max(axis=1)
        for t, threshold in enumerate(self.iou_thresholds):
            hit = best_iou >= threshold
            ignored[t] = hit & gt_difficult[best_gt]
            candidates = np.flatnonzero(hit & ~ignored[t])
            # 同一個 GT 只有分數最高（最先出現）的那個預測算 TP，其餘是重複偵測
            _, first = np.unique(best_gt[candidates], return_index=True)
            tp[t, candidates[first]] = True
        return tp, ignored

    def compute(self) -> Dict[str, float]:
        """回傳 mAP50、mAP50_95、mAP50_voc07（11 點插值），以及每類 AP50。"""
        num_classes = len(self.class_names)
        ap = np.full((len(self.iou_thresholds), num_classes), np.nan)
        ap07 = np.full(num_classes, np.nan)
        for c in range(1, num_classes + 1):
            npos = self._num_gt[c]
            if npos == 0:
                continue
            if not self._scores[c]:
                ap[:, c - 1] = 0.0
                ap07[c - 1] = 0.0
                continue
            scores = np.concatenate(self._scores[c])
            tp_all = np.concatenate(self._tp[c], axis=1)
            ignored_all = np.concatenate(self._ignored[c], axis=1)
            order = np.argsort(-scores, kind="stable")
            for t in range(len(self.iou_thresholds)):
                keep = ~ignored_all[t, order]
                tp = tp_all[t, order][keep].astype(np.float64)
                fp = 1.0 - tp
                tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
                recall = tp_cum / npos
                precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
                ap[t, c - 1] = _average_precision(recall, precision)
                if t == 0:
                    ap07[c - 1] = _average_precision_11point(recall, precision)

        metrics = {
            "mAP50": float(np.nanmean(ap[0])),
            "mAP50_95": float(np.nanmean(ap)),
            "mAP50_voc07": float(np.nanmean(ap07)),
        }
        for name, value in zip(self.class_names, ap[0]):
            metrics[f"AP50/{name}"] = float(value)
        return metrics


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC2010 之後的算法：精確率包絡線下的面積（所有點插值）。"""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _average_precision_11point(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC2007 的 11 點插值 AP。"""
    values = []
    for r in np.linspace(0, 1, 11):
        mask = recall >= r
        values.append(precision[mask].max() if mask.any() else 0.0)
    return float(np.mean(values))
