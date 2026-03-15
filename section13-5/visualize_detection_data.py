#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

def load_csv_data(csv_path):
    """
    CSVファイルを読み込み、DataFrameとして返す

    Args:
        csv_path (str): 読み込むCSVファイルのパス（timestampとdetection_countカラムを含む）

    Returns:
        DataFrame: 読み込んだデータ（エラー時はNone）
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def process_detection_data(df):
    """
    検出データを処理し、グラフ作成用に整形する

    Args:
        df (DataFrame): 生のCSVデータ（timestamp, detection_countカラムを含む）

    Returns:
        DataFrame: 整形後のデータ（timestamp列はdatetime型、detection_count列は数値型）
    """
    # タイムスタンプをdatetime型に変換（文字列から日時オブジェクトへ）
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 検出数が文字列の場合は数値に変換（数値変換できない値はNaNになる）
    if df['detection_count'].dtype == 'object':
        df['detection_count'] = pd.to_numeric(df['detection_count'], errors='coerce')

    # NaNを0で埋める（欠損データをゼロとして扱う）
    df['detection_count'] = df['detection_count'].fillna(0)

    return df

def create_detection_plot(df, output_path=None, show_plot=True):
    """
    検出数の時系列グラフを作成（3つのサブプロット構成）

    Args:
        df (DataFrame): 処理済みの検出データ（timestamp, detection_countカラムを含む）
        output_path (str): グラフ保存先のパス（Noneの場合は保存しない）
        show_plot (bool): グラフを画面表示するかどうか（デフォルト: True）

    Returns:
        Figure: 作成したmatplotlibのFigureオブジェクト
    """

    # 図の設定（3つのサブプロットを縦に配置）
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # 1. 検出数の時系列プロット（時間経過とともに検出数がどう変化したか）
    ax1.plot(df['timestamp'], df['detection_count'],
             marker='o', markersize=4, linewidth=1.5,
             color='#2E86AB', label='Detection Count')
    ax1.fill_between(df['timestamp'], 0, df['detection_count'],
                     alpha=0.3, color='#2E86AB')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Detection Count', fontsize=12)
    ax1.set_title('Insect Detection Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # X軸の時間フォーマット（1時間間隔で表示）
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. 累積検出数（観察開始からの合計検出数の推移）
    df['cumulative_detections'] = df['detection_count'].cumsum()
    ax2.plot(df['timestamp'], df['cumulative_detections'],
             linewidth=2, color='#A23B72', label='Cumulative Detections')
    ax2.fill_between(df['timestamp'], 0, df['cumulative_detections'],
                     alpha=0.3, color='#A23B72')

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Cumulative Count', fontsize=12)
    ax2.set_title('Cumulative Detection Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # X軸の時間フォーマット（1時間間隔で表示）
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. 時間帯別の活動量（1時間単位の棒グラフ）
    df['hour'] = df['timestamp'].dt.hour
    hourly_activity = df.groupby('hour')['detection_count'].sum()

    # 24時間分のデータを準備（存在しない時間は0で埋める）
    all_hours = pd.Series(index=range(24), data=0)
    all_hours.update(hourly_activity)

    # 21:00スタート、06:00終了になるように時間軸を再配置
    # 夜行性昆虫の観察に最適化（21:00-06:59の10時間を表示、07:00は含めない）
    # 理由: カブトムシは夜行性のため、活動時間帯を重点的に表示する
    start_hour = 21  # 観察開始時刻（21:00）
    end_hour = 7     # 観察終了時刻の次の時間（06:00の次は07:00だが、これは含めない）

    # 21,22,23,0,1,2,3,4,5,6の順番に時間を並び替え
    # range(21, 24)で21,22,23を取得し、range(0, 7)で0,1,2,3,4,5,6を取得して結合
    reordered_hours = list(range(start_hour, 24)) + list(range(0, end_hour))
    reordered_values = [all_hours[h] for h in reordered_hours]
    reordered_labels = [f'{h:02d}:00' for h in reordered_hours]

    # 10時間分のデータで棒グラフを作成
    bars = ax3.bar(range(len(reordered_hours)), reordered_values,
                   color='#F18F01', edgecolor='#C73E1D', linewidth=1.5, alpha=0.7)

    # 最大値のバーを強調表示（最も活動が活発だった時間帯を目立たせる）
    max_val = max(reordered_values)
    for i, val in enumerate(reordered_values):
        if val == max_val:
            bars[i].set_color('#C73E1D')
            break

    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Total Detections', fontsize=12)
    ax3.set_title('Activity Pattern by Hour (21:00 - 06:00)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(reordered_hours)))
    ax3.set_xticklabels(reordered_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 平均値ラインを追加（活動量の平均を可視化）
    mean_val = np.mean(reordered_values)
    ax3.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, label=f'Average: {mean_val:.1f}')
    ax3.legend(loc='upper right')

    # レイアウト調整（サブプロット間の余白を最適化）
    plt.tight_layout()

    # 統計情報をタイトルに追加
    total_detections = df['detection_count'].sum()
    max_detections = df['detection_count'].max()
    avg_detections = df['detection_count'].mean()
    observation_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600

    fig.suptitle(f'Total: {total_detections:.0f} detections | Max: {max_detections:.0f} | '
                f'Avg: {avg_detections:.2f} | Duration: {observation_duration:.1f} hours',
                fontsize=11, y=1.02)

    # ファイル保存（パスが指定されている場合）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {output_path}")

    # 画面表示（show_plotがTrueの場合）
    if show_plot:
        plt.show()

    return fig

def print_statistics(df):
    """
    検出データの統計情報をコンソールに表示

    Args:
        df (DataFrame): 処理済みの検出データ（timestamp, detection_countカラムを含む）
    """
    print("\n" + "="*50)
    print("Detection Statistics")
    print("="*50)

    # 基本統計情報の計算
    total_observations = len(df)  # 観察回数の合計
    total_detections = df['detection_count'].sum()  # 検出数の合計
    detections_with_insects = (df['detection_count'] > 0).sum()  # 昆虫が検出された回数
    detection_rate = (detections_with_insects / total_observations * 100) if total_observations > 0 else 0

    print(f"Total observations: {total_observations}")
    print(f"Total detections: {int(total_detections)}")
    print(f"Observations with detections: {detections_with_insects}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Average detections per observation: {df['detection_count'].mean():.2f}")
    print(f"Max detections in single observation: {int(df['detection_count'].max())}")

    # 時間範囲の表示
    if not df.empty:
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 3600  # 秒を時間に変換
        print(f"\nObservation period:")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(f"  Duration: {duration:.1f} hours")

    # 最も活発な時間帯の特定
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

    hourly_activity = df.groupby('hour')['detection_count'].sum()
    if not hourly_activity.empty:
        most_active_hour = hourly_activity.idxmax()  # 検出数が最大の時間帯
        print(f"\nMost active hour: {most_active_hour:02d}:00-{(most_active_hour+1)%24:02d}:00")
        print(f"  Detections in this hour: {int(hourly_activity.max())}")

    print("="*50)

def create_activity_heatmap(df, output_path=None, show_plot=True):
    """
    活動ヒートマップを作成（複数日のデータがある場合のみ有効）

    Args:
        df (DataFrame): 処理済みの検出データ（timestamp, detection_countカラムを含む）
        output_path (str): グラフ保存先のパス（Noneの場合は保存しない）
        show_plot (bool): グラフを画面表示するかどうか（デフォルト: True）

    Returns:
        Figure: 作成したmatplotlibのFigureオブジェク (データ不足の場合はNone)
    """

    # 日付と時間を分離（ヒートマップの縦横軸に使用）
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    # 日付×時間のピボットテーブル作成（各セルは検出数の合計値）
    pivot_data = df.pivot_table(values='detection_count',
                                index='hour',
                                columns='date',
                                aggfunc='sum',
                                fill_value=0)

    # データが1日分しかない場合はヒートマップ作成不可
    if pivot_data.empty or pivot_data.shape[1] == 1:
        print("Not enough data for heatmap (need multiple days)")
        return None

    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(12, 8))

    # imshowで色付きのマトリクスを表示（YlOrRd: 黄色から赤のカラーマップ）
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')

    # 軸設定（X軸: 日付、Y軸: 時刻）
    ax.set_xticks(range(pivot_data.shape[1]))
    ax.set_xticklabels([str(d) for d in pivot_data.columns], rotation=45, ha='right')
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(24)])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_title('Activity Heatmap', fontsize=14, fontweight='bold')

    # カラーバー追加（検出数の凡例）
    plt.colorbar(im, ax=ax, label='Detection Count')

    plt.tight_layout()

    # ファイル保存（元のファイル名の拡張子前に_heatmapを追加）
    if output_path:
        heatmap_path = output_path.replace('.png', '_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {heatmap_path}")

    # 画面表示（show_plotがTrueの場合）
    if show_plot:
        plt.show()

    return fig

def main():
    """
    メイン関数：コマンドライン引数を処理してデータ可視化を実行

    処理フロー:
    1. コマンドライン引数の解析
    2. CSVファイルの存在確認
    3. データ読み込みと前処理
    4. 統計情報の表示
    5. グラフ作成と保存
    6. オプションでヒートマップ作成
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="Visualize insect detection data from CSV files"
    )

    # 引数の定義
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('-o', '--output', default=None,
                       help='Output image file path (PNG format)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display the plot (only save)')
    parser.add_argument('--heatmap', action='store_true',
                       help='Also create activity heatmap (for multi-day data)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics without plotting')

    args = parser.parse_args()

    # CSVファイルの存在確認
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # データ読み込み
    df = load_csv_data(csv_path)
    if df is None or df.empty:
        print("Error: No data to process")
        sys.exit(1)

    # データ処理（型変換と欠損値処理）
    df = process_detection_data(df)

    # 統計情報表示
    print_statistics(df)

    # 統計情報のみ表示して終了（グラフ作成なし）
    if args.stats_only:
        return

    # 出力パス設定
    if args.output:
        output_path = Path(args.output)
    else:
        # デフォルトの出力パス（CSVと同じディレクトリに_graph.pngを追加）
        output_path = csv_path.parent / f"{csv_path.stem}_graph.png"

    # グラフ作成（--no-displayオプションがある場合は画面表示しない）
    show_plot = not args.no_display
    create_detection_plot(df, output_path, show_plot)

    # ヒートマップ作成（--heatmapオプションが指定されている場合のみ）
    if args.heatmap:
        create_activity_heatmap(df, output_path, show_plot)

    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
