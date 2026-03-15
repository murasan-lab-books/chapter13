#!/usr/bin/env python3
import argparse
import sys
import time
import csv
import json
from pathlib import Path
from datetime import datetime
import signal

try:
    from picamera2 import Picamera2
    from libcamera import controls
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    sys.exit(1)

# グローバル変数
picam2 = None
running = True
csv_writer = None
csv_file = None

def signal_handler(sig, frame):
    """
    Ctrl+C（SIGINT）シグナルを受信した際の終了処理ハンドラ

    ユーザーがCtrl+Cを押すと、このハンドラが呼ばれてプログラムを安全に終了します。
    グローバル変数runningをFalseにすることで、メインループを終了させます。
    """
    global running
    print("\nStopping logging test...")
    running = False

def distance_to_lens_position(distance_cm, max_lens=32.0):
    """
    距離(cm)をCamera Module 3 Wide用のレンズ位置パラメータに近似変換

    Args:
        distance_cm (float): フォーカスを合わせたい距離（cm単位、5cm～100cm）
        max_lens (float): レンズ位置の最大値（Camera Module 3 Wideでは32.0）

    Returns:
        float: レンズ位置パラメータ（0.0～32.0の範囲）
    """
    # 5cm以下の極近距離は最大レンズ位置
    if distance_cm <= 5:
        return max_lens
    # 100cm以上の遠距離はレンズ位置0.0（無限遠）
    elif distance_cm >= 100:
        return 0.0
    else:
        # 5cm～100cmの範囲は対数スケールで変換
        import math
        log_distance = math.log10(distance_cm / 5)
        lens_pos = max_lens * (1 - log_distance / math.log10(20))
        return max(0.0, min(max_lens, lens_pos))

def setup_logging(output_dir: Path):
    """
    ログファイル（CSVとJSONメタデータ）のセットアップ

    Args:
        output_dir (Path): ログファイルを保存するディレクトリパス

    Returns:
        tuple: (csv_path, metadata_path) - 作成したCSVファイルとメタデータファイルのパス
    """
    global csv_writer, csv_file

    # 出力ディレクトリを作成（存在しない場合）
    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ付きファイル名を生成（YYYYMMDD_HHMMSS形式）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"left_half_detection_log_{timestamp}.csv"
    metadata_path = output_dir / f"left_half_metadata_{timestamp}.json"

    # CSVファイル作成（UTF-8エンコーディング）
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # ヘッダー行を書き込み（検出エリア情報を含む全カラム定義）
    csv_writer.writerow([
        'timestamp', 'observation_number', 'detection_count', 'has_detection',
        'class_names', 'confidence_values', 'bbox_coordinates',
        'center_x', 'center_y', 'bbox_width', 'bbox_height', 'area',
        'detection_area', 'processing_time_ms', 'image_saved', 'image_filename'
    ])
    csv_file.flush()  # ヘッダーをすぐにファイルに書き込み

    # メタデータファイル作成（観測セッションの設定情報を記録）
    metadata = {
        'start_time': datetime.now().isoformat(),
        'detection_area': 'left_half',
        'area_description': 'Only left 50% of camera view is monitored',
        'log_file': str(csv_path),
        'system_info': {
            'camera': 'Camera Module 3 Wide NoIR',
            'platform': 'Raspberry Pi'
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return csv_path, metadata_path

def save_detection_to_csv(observation_num, detections, processing_time, image_saved=False, image_filename=None):
    """
    検出結果を1行のCSVレコードとして保存

    Args:
        observation_num (int): 観測番号（何回目の観測か）
        detections (list): 検出結果のリスト（各要素は検出情報の辞書）
        processing_time (float): 処理時間（ミリ秒）
        image_saved (bool): 画像を保存したかどうか（デフォルト: False）
        image_filename (str): 保存した画像のファイル名（デフォルト: None）
    """
    global csv_writer, csv_file

    # CSVライターが初期化されていない場合は何もしない
    if not csv_writer:
        return

    # タイムスタンプと検出有無を記録
    timestamp = datetime.now().isoformat()
    detection_count = len(detections) if detections else 0
    has_detection = detection_count > 0

    # 検出があった場合、各検出情報を文字列に変換
    if has_detection:
        # クラス名のリストを作成
        class_names = [d['class'] for d in detections]
        # 信頼度を小数点3桁の文字列に変換
        confidence_values = [f"{d['confidence']:.3f}" for d in detections]
        # バウンディングボックスの座標を "(x1,y1,x2,y2)" 形式に変換
        bbox_coords = [f"({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})" for d in detections]

        # バウンディングボックスの中心座標と寸法を抽出
        centers_x = [f"{d['center_x']:.1f}" for d in detections]
        centers_y = [f"{d['center_y']:.1f}" for d in detections]
        widths = [f"{d['width']:.1f}" for d in detections]
        heights = [f"{d['height']:.1f}" for d in detections]
        areas = [f"{d['area']:.1f}" for d in detections]

        # 複数検出がある場合はセミコロン(;)で結合
        class_names_str = ';'.join(class_names)
        confidence_str = ';'.join(confidence_values)
        bbox_str = ';'.join(bbox_coords)
        center_x_str = ';'.join(centers_x)
        center_y_str = ';'.join(centers_y)
        width_str = ';'.join(widths)
        height_str = ';'.join(heights)
        area_str = ';'.join(areas)
    else:
        # 検出がない場合は全て空文字列
        class_names_str = ''
        confidence_str = ''
        bbox_str = ''
        center_x_str = ''
        center_y_str = ''
        width_str = ''
        height_str = ''
        area_str = ''

    # CSVの1行分のデータを作成
    row = [
        timestamp,
        observation_num,
        detection_count,
        has_detection,
        class_names_str,
        confidence_str,
        bbox_str,
        center_x_str,
        center_y_str,
        width_str,
        height_str,
        area_str,
        'left_half',  # 検出エリア情報（画面左半分）
        f"{processing_time:.1f}",
        image_saved,
        image_filename or ''
    ]

    # CSVファイルに書き込み
    csv_writer.writerow(row)
    csv_file.flush()  # バッファをすぐにファイルに書き込む

def test_logging_left_half(
    model_path: str = '../weights/best.pt',
    confidence: float = 0.3,
    width: int = 2304,
    height: int = 1296,
    focus_distance: float = 20.0,
    interval: int = 10,
    duration: int = 60,
    save_images: bool = False,
    output_dir: str = None,
    show_boundary: bool = False,
    exposure_value: float = -0.5,
    contrast: float = 2.0,
    brightness: float = 0.0
):
    """
    画面左半分エリアのみを検出対象とする長時間ロギング機能

    Args:
        model_path (str): YOLOv8モデルファイルのパス（デフォルト: ../weights/best.pt）
        confidence (float): 検出の信頼度閾値（0.0～1.0、デフォルト: 0.3）
        width (int): カメラ解像度の横幅（デフォルト: 2304）
        height (int): カメラ解像度の縦幅（デフォルト: 1296）
        focus_distance (float): フォーカス距離（cm単位、0で自動フォーカス、デフォルト: 20.0）
        interval (int): 観測間隔（秒単位、デフォルト: 10）
        duration (int): 観測継続時間（秒単位、0で無制限、デフォルト: 60）
        save_images (bool): 検出時に画像を保存するか（デフォルト: False）
        output_dir (str): ログファイルの出力ディレクトリ（Noneで自動設定、デフォルト: None）
        show_boundary (bool): 画像保存時に左半分境界線を描画するか（デフォルト: False）
        exposure_value (float): 露出補正値（-8.0～8.0、負で暗く、デフォルト: -0.5）
        contrast (float): コントラスト値（0.0～32.0、デフォルト: 2.0）
        brightness (float): 明るさ値（-1.0～1.0、デフォルト: 0.0）

    Returns:
        bool: 処理が正常に完了したらTrue、エラーが発生したらFalse
    """
    global picam2, running

    # デフォルトの出力ディレクトリを設定
    # output_dirが指定されていない場合は、スクリプトと同じディレクトリ内のinsect_detection_logsを使用
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / "insect_detection_logs"
    else:
        output_path = Path(output_dir)

    # ログファイル（CSV、JSON）を作成
    csv_path, metadata_path = setup_logging(output_path)

    # YOLOv8モデルの読み込み
    print(f"\nLoading model: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"Model loaded. Classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Picamera2（Camera Module 3 Wide）の初期化
    print(f"\nInitializing Picamera2...")
    try:
        picam2 = Picamera2()

        # カメラセンサーの解像度を取得して表示
        sensor_resolution = picam2.sensor_resolution
        print(f"Sensor resolution: {sensor_resolution}")
        print(f"Using binned sensor mode: {width}x{height} for full wide-angle coverage")
        print(f"Detection area: LEFT HALF ONLY (0 to {width//2} pixels)")

        # カメラ設定を作成（RGB888形式で指定解像度を取得）
        config = picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=4  # バッファ数（フレーム取得の安定性向上）
        )
        picam2.configure(config)

        # カメラを起動して安定化のため2秒待機
        print("Starting camera...")
        picam2.start()
        time.sleep(2)  # カメラが安定するまで待機

        # フォーカス設定（オートフォーカスまたはマニュアルフォーカス）
        # レンズ位置制御パラメータの範囲を取得
        lens_controls = picam2.camera_controls.get("LensPosition")
        if lens_controls:
            lp_min, lp_max, lp_default = lens_controls
        else:
            # 取得できない場合はCamera Module 3のデフォルト値を使用
            lp_min, lp_max, lp_default = 0.0, 32.0, 1.0

        if focus_distance == 0:
            # オートフォーカスモード（距離0の場合）
            print(f"Setting auto focus mode...")
            picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
            print("Auto focus mode enabled")
            time.sleep(2.0)  # オートフォーカスの調整時間を確保
        else:
            # マニュアルフォーカスモード（指定距離にフォーカス）
            print(f"Setting manual focus for {focus_distance}cm...")
            target_lens_pos = distance_to_lens_position(focus_distance, lp_max)
            print(f"Target lens position: {target_lens_pos:.1f}")
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual})
            time.sleep(0.5)
            picam2.set_controls({"LensPosition": float(target_lens_pos)})
            time.sleep(1.0)  # レンズが目標位置に移動するまで待機

        # 露出補正の設定（デフォルト以外の値が指定された場合）
        if exposure_value != 0.0:
            print(f"Setting exposure compensation: {exposure_value}")
            picam2.set_controls({"ExposureValue": exposure_value})

        # コントラストの設定（デフォルト1.0以外の値が指定された場合）
        if contrast != 1.0:
            print(f"Setting contrast: {contrast}")
            picam2.set_controls({"Contrast": contrast})

        # 明るさの設定（デフォルト0.0以外の値が指定された場合）
        if brightness != 0.0:
            print(f"Setting brightness: {brightness}")
            picam2.set_controls({"Brightness": brightness})

    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

    # メタデータファイルに観測パラメータを追記
    # 既存のメタデータファイルを読み込み
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 観測設定パラメータを追加
    metadata.update({
        'focus_mode': 'auto' if focus_distance == 0 else 'manual',
        'focus_distance_cm': focus_distance if focus_distance > 0 else None,
        'model_path': model_path,
        'confidence_threshold': confidence,
        'resolution': f"{width}x{height}",
        'detection_width': width // 2,  # 左半分のみなので横幅の半分
        'interval_seconds': interval,
        'duration_seconds': duration,
        'save_images': save_images
    })

    # 更新したメタデータをファイルに書き戻し
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*50)
    print("Starting LEFT HALF detection logging")
    print(f"Detection area: 0 to {width//2} pixels (left half)")
    print(f"Interval: {interval} seconds")
    print(f"Duration: {duration} seconds")
    print(f"Save images: {save_images}")
    print("="*50 + "\n")

    # Ctrl+C（SIGINT）シグナルを受信した時のハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

    # 検出時の画像保存用ディレクトリを作成
    if save_images:
        script_dir = Path(__file__).parent
        images_dir = script_dir / "images"
        images_dir.mkdir(exist_ok=True)  # ディレクトリが存在しない場合のみ作成
        print(f"Images will be saved to: {images_dir}")

    # 観測統計用の変数を初期化
    observation_count = 0  # 観測回数カウンター
    start_time = time.time()  # 開始時刻を記録
    total_detections = 0  # 総検出数カウンター

    # 左半分と右半分の境界線のX座標（画像の中心）
    boundary_x = width // 2

    try:
        # メイン観測ループ（runningがFalseになるまで継続）
        while running:
            obs_start = time.time()  # この観測の開始時刻

            # カメラからフレームを取得
            frame = picam2.capture_array()
            if frame is None:
                # フレーム取得に失敗した場合はスキップ
                print(f"[WARNING] Frame capture failed, skipping...")
                time.sleep(1)
                continue

            observation_count += 1

            # 画像の左半分のみを切り出す（0からboundary_xまで）
            # NumPy配列のスライス：frame[縦方向, 横方向]
            left_half_frame = frame[:, :boundary_x]

            # YOLOv8で物体検出を実行（左半分の画像のみを対象）
            results = model.predict(
                source=left_half_frame,
                device='cpu',  # CPUで推論
                conf=confidence,  # 信頼度閾値
                verbose=False  # 詳細ログを抑制
            )

            # 処理時間を計測（ミリ秒単位）
            processing_time = (time.time() - obs_start) * 1000

            # 検出結果を解析してリスト化
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 左半分での座標なので、そのまま使用
                    # （元の画像での座標は同じ）

                    # バウンディングボックスの中心座標と寸法を計算
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    area = bbox_width * bbox_height

                    # 検出情報を辞書にまとめる
                    detection = {
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': bbox_width,
                        'height': bbox_height,
                        'area': area
                    }
                    detections.append(detection)
                    total_detections += 1

            # 検出時の画像保存処理（オプション機能）
            image_saved = False
            image_filename = None
            if save_images and detections:
                # タイムスタンプ付きファイル名を生成（ミリ秒まで含む）
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                image_filename = f"left_half_detection_{timestamp}.jpg"
                image_path = images_dir / image_filename

                # 元のフレームをコピーして検出結果を描画
                annotated_frame = frame.copy()

                # 左半分境界線を描画（オプション）
                if show_boundary:
                    # 境界線を青色で描画
                    cv2.line(annotated_frame, (boundary_x, 0), (boundary_x, height),
                            (255, 0, 0), 2)
                    # "Detection Area"テキストを黄色で表示
                    cv2.putText(annotated_frame, "Detection Area", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # 各検出結果をバウンディングボックスとラベルで描画
                for det in detections:
                    x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                    # バウンディングボックスを緑色で描画
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # クラス名と信頼度をラベルとして表示
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # RGBからBGRに変換してJPEGファイルとして保存
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), annotated_frame_bgr)
                image_saved = True

            # 検出結果をCSVファイルに記録
            try:
                save_detection_to_csv(
                    observation_count,
                    detections,
                    processing_time,
                    image_saved,
                    image_filename
                )
            except Exception as csv_error:
                print(f"[ERROR] Failed to save CSV: {csv_error}")

            # 検出結果をコンソールに表示
            if detections:
                # 各検出のクラス名、信頼度、位置を文字列化
                detection_list = []
                for d in detections:
                    pos_info = f"@({d['center_x']:.0f},{d['center_y']:.0f})"
                    detection_list.append(f"{d['class']}({d['confidence']:.2f}){pos_info}")
                detection_str = ', '.join(detection_list)
                print(f"[{observation_count:04d}] LEFT HALF: {len(detections)} detections: {detection_str} | {processing_time:.1f}ms")
            else:
                print(f"[{observation_count:04d}] LEFT HALF: No detections | {processing_time:.1f}ms")

            # 観測継続時間のチェック
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                # 指定時間に到達したら終了
                print(f"\nDuration completed ({duration} seconds)")
                break

            # 次の観測まで待機（インターバル時間）
            if running and interval > 0:
                time.sleep(interval)

    except Exception as e:
        print(f"\nError during logging: {e}")
        return False

    finally:
        # リソースのクリーンアップ（必ず実行される）
        print("\nCleaning up...")
        if picam2:
            # カメラを停止して解放
            picam2.stop()
            picam2.close()

        if csv_file:
            # CSVファイルを閉じる
            csv_file.close()

        # 観測統計サマリーを表示
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print("Left Half Detection Summary")
        print("="*50)
        print(f"Total observations: {observation_count}")
        print(f"Total detections (left half): {total_detections}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        if observation_count > 0:
            print(f"Average detections per observation: {total_detections/observation_count:.2f}")
        print(f"Log file: {csv_path}")
        print("="*50)

    return True

def main():
    """
    メイン関数：コマンドライン引数を処理して左半分検出ロギングを実行

    処理フロー：
    1. コマンドライン引数の解析（モデルパス、信頼度、解像度、フォーカス設定など）
    2. 引数の値を表示してユーザーに確認
    3. test_logging_left_half()関数を呼び出して長時間観測を実行
    4. 成功/失敗に応じた終了コードを返す
    """
    # コマンドライン引数パーサーの設定
    parser = argparse.ArgumentParser(
        description="Left half detection logging with Picamera2 and YOLOv8"
    )

    # 各コマンドライン引数の定義
    parser.add_argument('--model', default='../weights/best.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--width', type=int, default=2304, help='Width (default: 2304)')
    parser.add_argument('--height', type=int, default=1296, help='Height (default: 1296)')
    parser.add_argument('--distance', type=float, default=20.0,
                       help='Focus distance in cm (use 0 for auto focus)')
    parser.add_argument('--auto-focus', action='store_true',
                       help='Enable auto focus mode')
    parser.add_argument('--interval', type=int, default=10,
                       help='Observation interval in seconds')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (0 for unlimited)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save detection images')
    parser.add_argument('--show-boundary', action='store_true',
                       help='Show boundary line in saved images')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for logs')
    parser.add_argument('--exposure', type=float, default=-0.5,
                       help='Exposure compensation (-8.0 to 8.0, negative for darker)')
    parser.add_argument('--contrast', type=float, default=2.0,
                       help='Contrast (0.0 to 32.0, default 2.0)')
    parser.add_argument('--brightness', type=float, default=0.0,
                       help='Brightness (-1.0 to 1.0, default 0.0)')

    # コマンドライン引数を解析
    args = parser.parse_args()

    # 観測パラメータの表示（ユーザーが設定を確認できるように）
    print("\nLeft Half Detection Logging Test")
    print("="*50)
    print(f"Model: {args.model}")
    # --auto-focusフラグが指定されている場合は距離0（オートフォーカス）
    focus_distance = 0 if args.auto_focus else args.distance
    focus_mode = "Auto" if focus_distance == 0 else f"{focus_distance}cm"
    print(f"Focus mode: {focus_mode}")
    print(f"Detection area: LEFT HALF ONLY")
    print(f"Confidence: {args.conf}")
    print(f"Interval: {args.interval}s")
    print(f"Duration: {args.duration}s")
    print()

    # 左半分検出ロギング処理を実行
    success = test_logging_left_half(
        model_path=args.model,
        confidence=args.conf,
        width=args.width,
        height=args.height,
        focus_distance=focus_distance,
        interval=args.interval,
        duration=args.duration,
        save_images=args.save_images,
        output_dir=args.output_dir,
        show_boundary=args.show_boundary,
        exposure_value=args.exposure,
        contrast=args.contrast,
        brightness=args.brightness
    )

    # 終了コードを返す（成功: 0、失敗: 1）
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
