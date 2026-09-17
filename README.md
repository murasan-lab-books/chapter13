# 第13章 AIカメラで昆虫の動きをトラッキングして自動観察システムを作ろう

本章では、昆虫検出AIモデルを学習し、Raspberry Piのカメラで昆虫の活動を記録・可視化するシステムを作ります。

## 収録内容

- `section13-2/train_yolo.py`：昆虫検出モデルを学習するスクリプト
- `section13-2/detect_insect.py`：昆虫を検出するスクリプト
- `section13-3/production_camera_left_half_realtime.py`：本番環境で検出モデルをテストするスクリプト
- `section13-4/production_logging_left_half.py`：検出結果を記録するスクリプト
- `section13-5/visualize_detection_data.py`：記録したデータをグラフで可視化するスクリプト

## データセットの出典

昆虫検出モデルの学習には、[Beetle Dataset](https://universe.roboflow.com/z-algae-bilby/beetle)（CC BY 4.0）を使用しています。外部に公開・頒布する場合は、出典とライセンス情報を明記してください。

## ライセンス

サンプルコードはAGPL-3.0ライセンスで提供しています。詳細はLICENSEファイルを参照してください。
