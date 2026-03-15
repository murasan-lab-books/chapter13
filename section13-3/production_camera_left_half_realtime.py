#!/usr/bin/env python3

import argparse
import sys
import time
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

def signal_handler(sig, frame):
    """
    Ctrl+C割り込み信号を受信してプログラムを安全に終了する

    グローバル変数runningをFalseに設定することで、メインループを停止させる。
    カメラリソースの解放はfinally節で確実に実行される。

    Args:
        sig: シグナル番号（通常はSIGINT）
        frame: 現在のスタックフレーム（未使用）
    """
    global running
    print("\nStopping camera test...")
    running = False

def distance_to_lens_position(distance_cm, max_lens=32.0):
    """
    距離(cm)をCamera Module 3 Wide用のレンズ位置に変換

    対数スケールを使用してピント距離をレンズ位置パラメータに変換する。
    5cm以下は最大値、100cm以上は最小値として扱う。

    Args:
        distance_cm (float): ピント合わせ距離（cm単位、5-100cmの範囲を想定）
        max_lens (float): レンズ位置の最大値（Camera Module 3 Wideの場合は32.0）

    Returns:
        float: レンズ位置パラメータ（0.0-max_lensの範囲）
    """
    if distance_cm <= 5:
        return max_lens
    elif distance_cm >= 100:
        return 0.0
    else:
        import math
        log_distance = math.log10(distance_cm / 5)
        lens_pos = max_lens * (1 - log_distance / math.log10(20))
        return max(0.0, min(max_lens, lens_pos))

def test_camera_left_half(
    model_path: str = '../weights/best.pt',
    confidence: float = 0.3,
    width: int = 2304,
    height: int = 1296,
    show_display: bool = True,
    focus_distance: float = 20.0,
    display_scale: float = 0.5,
    exposure_value: float = -0.5,
    contrast: float = 2.0,
    brightness: float = 0.0
):
    """
    画面左半分のみを検出対象としてリアルタイム表示する

    Camera Module 3 Wideで撮影した画像の左半分のみにYOLOv8を適用し、
    検出結果をリアルタイムで画面表示する。検出エリアと無視エリアを
    視覚的に区別して表示する。

    Args:
        model_path (str): YOLOv8モデルファイルのパス（.pt形式）
        confidence (float): 検出の信頼度閾値（0.0-1.0、通常0.3）
        width (int): カメラ解像度の幅（Camera Module 3 Wideは2304推奨）
        height (int): カメラ解像度の高さ（Camera Module 3 Wideは1296推奨）
        show_display (bool): 画面表示の有無（Falseでヘッドレスモード）
        focus_distance (float): ピント距離（cm、0でオートフォーカス）
        display_scale (float): 表示ウィンドウのスケール（0.5で半分サイズ）
        exposure_value (float): 露出補正（-8.0 - 8.0、負の値で暗く）
        contrast (float): コントラスト（0.0 - 32.0、デフォルト2.0）
        brightness (float): 明るさ（-1.0 - 1.0、デフォルト0.0）

    Returns:
        bool: 処理が正常終了した場合True、エラー時False
    """
    global picam2, running

    # YOLOv8モデルの読み込み
    # Ultralyticsライブラリを使用して学習済みモデル（.ptファイル）をロード
    print(f"\nLoading model: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully. Classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Picamera2の初期化
    # Raspberry Pi公式のPicamera2ライブラリを使用してカメラを制御
    print(f"\nInitializing Picamera2...")
    try:
        picam2 = Picamera2()

        # カメラプロパティの取得
        # 接続されているカメラモジュールの情報（モデル名など）を確認
        camera_properties = picam2.camera_properties
        print(f"Camera Model: {camera_properties.get('Model', 'Unknown')}")

        # LensPosition範囲の確認
        # カメラモジュールによってフォーカス制御の範囲が異なるため確認が必要
        # Camera Module 3 Wideの場合は通常0.0-32.0の範囲
        lens_controls = picam2.camera_controls.get("LensPosition")
        if lens_controls:
            lp_min, lp_max, lp_default = lens_controls
            print(f"LensPosition range: {lp_min} - {lp_max} (default: {lp_default})")
        else:
            lp_min, lp_max, lp_default = 0.0, 32.0, 1.0
            print("Warning: Using assumed LensPosition range: 0-32")

        # カメラ解像度の設定
        # センサーの物理解像度を確認（Camera Module 3 Wideは4608x2592）
        sensor_resolution = picam2.sensor_resolution
        print(f"Sensor resolution: {sensor_resolution}")
        print(f"Using binned sensor mode: {width}x{height} for full wide-angle coverage")
        print(f"Detection area: LEFT HALF ONLY (0 to {width//2} pixels)")

        # カメラ設定の作成
        # RGB888形式で指定解像度の映像を取得する設定
        # buffer_count=4でフレームバッファを4枚確保（処理遅延を防ぐ）
        config = picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=4
        )
        picam2.configure(config)

        # カメラの起動
        # 設定を適用してカメラストリームを開始
        # 2秒待機してカメラが安定するのを待つ
        print("Starting camera...")
        picam2.start()
        time.sleep(2)

        # フォーカス設定
        # オートフォーカスまたはマニュアルフォーカスを選択
        if focus_distance == 0:
            # オートフォーカスモード
            # カメラが自動的に最適なピント位置を探す
            print(f"\nSetting auto focus mode...")
            picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
            print("Auto focus mode enabled")
            time.sleep(2.0)
        else:
            # マニュアルフォーカスモード
            # 指定された距離（cm）に基づいてレンズ位置を計算・設定
            print(f"\nSetting manual focus for {focus_distance}cm distance...")
            target_lens_pos = distance_to_lens_position(focus_distance, lp_max)
            print(f"Target lens position: {target_lens_pos:.1f}")
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual})
            time.sleep(0.5)
            picam2.set_controls({"LensPosition": float(target_lens_pos)})
            time.sleep(1.0)

        # 露出・コントラスト・明るさの設定
        # 撮影環境に応じて画質を調整
        if exposure_value != 0.0:
            # 露出補正：負の値で暗く、正の値で明るく（-8.0〜8.0）
            print(f"Setting exposure compensation: {exposure_value}")
            picam2.set_controls({"ExposureValue": exposure_value})

        if contrast != 1.0:
            # コントラスト：明暗の差を調整（0.0〜32.0、1.0が標準）
            print(f"Setting contrast: {contrast}")
            picam2.set_controls({"Contrast": contrast})

        if brightness != 0.0:
            # 明るさ：全体的な明度を調整（-1.0〜1.0、0.0が標準）
            print(f"Setting brightness: {brightness}")
            picam2.set_controls({"Brightness": brightness})

    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

    print("\n" + "="*50)
    print("Starting LEFT HALF detection test")
    print(f"Detection area: Left half only (0-{width//2} pixels)")
    print("Press 'q' to quit, 's' to save image")
    print("="*50 + "\n")

    # Ctrl+Cシグナルハンドラの設定
    # ユーザーがCtrl+Cを押した時にsignal_handler関数を呼び出す
    signal.signal(signal.SIGINT, signal_handler)

    # 画像保存ディレクトリの準備
    # 's'キーで保存する画像の保存先ディレクトリを作成
    script_dir = Path(__file__).parent
    images_dir = script_dir / "images"
    images_dir.mkdir(exist_ok=True)  # 存在しない場合のみ作成

    # カウンター変数の初期化
    frame_count = 0           # 処理したフレーム数
    total_detections = 0      # 検出した昆虫の総数
    boundary_x = width // 2   # 左半分の境界線のX座標（画面の中央）

    try:
        while running:
            # カメラからフレームを取得
            # Picamera2のcapture_array()でRGB配列として画像を取得
            frame = picam2.capture_array()
            if frame is None:
                continue

            frame_count += 1

            # 左半分のみを切り出し
            # NumPy配列のスライシングを使用（全行、0列目からboundary_x列目まで）
            left_half_frame = frame[:, :boundary_x]

            # YOLOv8による物体検出（左半分のみ）
            # CPU推論、指定した信頼度閾値以上の検出結果のみ取得
            results = model.predict(
                source=left_half_frame,
                device='cpu',           # CPU推論（Raspberry Piでの動作）
                conf=confidence,        # 信頼度閾値（通常0.3）
                verbose=False           # 詳細ログを抑制
            )

            # 表示用フレームの準備（元の画像全体をコピー）
            # copy()で元のフレームを保持しつつ、検出結果を重ねて描画
            display_frame = frame.copy()

            # 境界線の描画（縦線で左半分と右半分を視覚的に区別）
            # RGB形式なので(255, 0, 0)は赤。ただし表示前にBGR変換するため
            # 画面上では青く表示される
            cv2.line(display_frame, (boundary_x, 0), (boundary_x, height),
                    (255, 0, 0), 2)

            # 左半分の枠線（緑色の矩形で検出エリアを強調）
            cv2.rectangle(display_frame, (0, 0), (boundary_x-1, height-1),
                         (0, 255, 0), 2)

            # エリアラベルの描画
            # 左半分：検出対象エリア（緑色）
            # 右半分：無視エリア（グレー）
            cv2.putText(display_frame, "Detection Area", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Ignored Area", (boundary_x + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

            # 検出結果の処理
            # YOLOv8の検出結果から各物体の情報を取り出して描画
            detections = []
            if results[0].boxes is not None:
                # 検出された各物体（バウンディングボックス）を処理
                for box in results[0].boxes:
                    # バウンディングボックスの座標を取得（左上と右下）
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls)          # クラスID（昆虫の種類）
                    conf = float(box.conf)      # 信頼度スコア

                    # バウンディングボックスの描画（緑色の矩形）
                    # 座標は左半分画像での座標なのでそのまま使用
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 255, 0), 2)

                    # クラス名と信頼度のラベル作成
                    label = f"{model.names[cls]} {conf:.2f}"
                    # ラベルのサイズを取得（背景矩形のサイズ計算用）
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    # ラベル背景（緑色で塗りつぶし）
                    cv2.rectangle(display_frame,
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)),
                                (0, 255, 0), -1)
                    # ラベルテキスト（黒色）
                    cv2.putText(display_frame, label, (int(x1), int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # 検出情報をリストに追加
                    detections.append({
                        'class': model.names[cls],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
                    total_detections += 1

            # ステータス表示
            status_text = f"Frame: {frame_count} | Detections: {len(detections)}"
            cv2.putText(display_frame, status_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # コンソール出力
            if len(detections) > 0:
                detection_list = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                print(f"Frame {frame_count}: {len(detections)} detections in LEFT HALF - {', '.join(detection_list)}")

            # 画面表示
            if show_display:
                # RGBからBGRに色空間を変換
                # OpenCVのimshow()はBGR形式を期待するため変換が必要
                display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # ウィンドウサイズの調整（リサイズ）
                # 元の解像度が高すぎる場合に画面サイズに合わせて縮小
                if display_scale != 1.0:
                    display_width = int(width * display_scale)
                    display_height = int(height * display_scale)
                    display_frame_resized = cv2.resize(display_frame_bgr, (display_width, display_height))
                    cv2.imshow('Left Half Detection Test', display_frame_resized)
                else:
                    cv2.imshow('Left Half Detection Test', display_frame_bgr)

                # キー入力の処理
                # waitKey(1)で1msだけキー入力を待機（リアルタイム表示を維持）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # 'q'キーで終了
                    print("Quit requested")
                    break
                elif key == ord('s'):
                    # 's'キーで現在のフレームを画像ファイルとして保存
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"left_half_test_{timestamp}.jpg"
                    filepath = images_dir / filename
                    cv2.imwrite(str(filepath), display_frame_bgr)
                    print(f"Image saved: {filepath}")

            # CPU使用率を抑える
            # 10ms待機してCPU負荷を軽減（Raspberry Piでの安定動作のため）
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Ctrl+Cによる割り込み
        print("\nInterrupted by user")
    except Exception as e:
        # その他のエラー
        print(f"Error during detection: {e}")
        return False
    finally:
        # リソースのクリーンアップ（必ず実行）
        # カメラリソースとウィンドウを確実に解放
        print("\nCleaning up...")
        if show_display:
            cv2.destroyAllWindows()  # OpenCVウィンドウを閉じる
        if picam2:
            picam2.stop()            # カメラストリームを停止
            picam2.close()           # カメラリソースを解放

        # 処理結果のサマリー表示
        print(f"\nTest completed")
        print(f"Total frames: {frame_count}")
        print(f"Total detections (left half): {total_detections}")
        if frame_count > 0:
            print(f"Average detections per frame: {total_detections/frame_count:.2f}")

    return True

def main():
    """
    メイン関数：コマンドライン引数を解析して左半分検出テストを実行

    処理フロー:
    1. コマンドライン引数の定義と解析
    2. フォーカスモードの決定（オートまたはマニュアル）
    3. 設定内容のコンソール表示
    4. test_camera_left_half関数の実行
    5. 結果に応じた終了コードの返却
    """
    # コマンドライン引数パーサーの設定
    parser = argparse.ArgumentParser(
        description="Test left half detection with real-time display"
    )

    # 引数の定義
    # 各パラメータはデフォルト値を持ち、ユーザーは必要に応じて上書き可能
    parser.add_argument('--model', default='../weights/best.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--width', type=int, default=2304, help='Width (default: 2304)')
    parser.add_argument('--height', type=int, default=1296, help='Height (default: 1296)')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--distance', type=float, default=20.0,
                       help='Focus distance in cm (5-100), use 0 for auto focus')
    parser.add_argument('--auto-focus', action='store_true',
                       help='Enable auto focus mode')
    parser.add_argument('--display-scale', type=float, default=0.5,
                       help='Display window scale (0.5 = half size, 1.0 = full size)')
    parser.add_argument('--exposure', type=float, default=-0.5,
                       help='Exposure compensation (-8.0 to 8.0, negative for darker)')
    parser.add_argument('--contrast', type=float, default=2.0,
                       help='Contrast (0.0 to 32.0, default 2.0)')
    parser.add_argument('--brightness', type=float, default=0.0,
                       help='Brightness (-1.0 to 1.0, default 0.0)')

    # 引数の解析
    args = parser.parse_args()

    # 設定内容のコンソール表示
    # ユーザーが指定したパラメータを確認できるよう表示
    print("\nLeft Half Detection Real-time Test")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"Resolution: {args.width}x{args.height}")

    # フォーカスモードの決定
    # --auto-focusフラグまたは--distance 0でオートフォーカス
    focus_distance = 0 if args.auto_focus else args.distance
    focus_mode = "Auto" if focus_distance == 0 else f"{focus_distance}cm"
    print(f"Focus: {focus_mode}")
    print(f"Display: {'No' if args.no_display else 'Yes'}")
    print()

    # 左半分検出テストの実行
    # 全パラメータを関数に渡して処理を開始
    success = test_camera_left_half(
        model_path=args.model,
        confidence=args.conf,
        width=args.width,
        height=args.height,
        show_display=not args.no_display,
        focus_distance=focus_distance,
        display_scale=args.display_scale,
        exposure_value=args.exposure,
        contrast=args.contrast,
        brightness=args.brightness
    )

    # 終了コードの返却（0: 成功、1: 失敗）
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
