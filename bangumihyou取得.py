import os
import time
import requests
import subprocess
from datetime import datetime, timedelta

# ==========================================
# ⚙️ 設定エリア
# ==========================================
START_DATE = "2023-04-01"
END_DATE   = "2024-04-01"

# 🌟 変更点：保存先をローカルのDドライブに変更（\でエラーが起きないよう r をつけています）
BASE_DIR = r"D:\MONSTER AI"

# ==========================================
# 🚀 ダウンロード＆解凍関数 (デバッグ強化版)
# ==========================================
def download_and_extract_custom(start_str, end_str):
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    print(f"📥 mbrace 全自動スクレイピング開始 (調査モード)")
    print("-" * 50)

    current_date = start_date

    while current_date <= end_date:
        yyyy_mm = current_date.strftime("%Y%m")
        yy_mm_dd = current_date.strftime("%y%m%d")

        b_dir = os.path.join(BASE_DIR, "bangumihyou", yyyy_mm)
        k_dir = os.path.join(BASE_DIR, "kyousouseiseki", yyyy_mm)
        os.makedirs(b_dir, exist_ok=True)
        os.makedirs(k_dir, exist_ok=True)

        targets = [
            {'type': '番組表(B)', 'url_base': 'https://www1.mbrace.or.jp/od2/B', 'prefix': 'b', 'save_dir': b_dir},
            {'type': '競争成績(K)', 'url_base': 'https://www1.mbrace.or.jp/od2/K', 'prefix': 'k', 'save_dir': k_dir}
        ]

        for target in targets:
            file_name = f"{target['prefix']}{yy_mm_dd}.lzh"
            download_url = f"{target['url_base']}/{yyyy_mm}/{file_name}"
            lzh_save_path = os.path.join(target['save_dir'], file_name)
            txt_name = f"{target['prefix'].upper()}{yy_mm_dd}.TXT"
            txt_save_path = os.path.join(target['save_dir'], txt_name)

            if os.path.exists(txt_save_path):
                print(f"⏭️ 取得済スキップ: {txt_name}")
                continue

            try:
                # 📥 1. ダウンロード
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(lzh_save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk: f.write(chunk)

                    # ダウンロードされたファイルのサイズを確認（小さすぎるとエラーページの可能性）
                    file_size = os.path.getsize(lzh_save_path)
                    if file_size < 1000:
                        print(f"❌ 失敗: ダウンロードしたファイルが異常に小さいです ({file_size} bytes)。アクセス拒否されている可能性があります。")
                        continue

                    # 📦 2. 7-Zipで解凍 (エラー出力をキャプチャ)
                    # ⚠️ Windows環境で '7z' が認識されない場合はフルパスに変更してください (例: r'C:\Program Files\7-Zip\7z.exe')
                    # 📦 2. 7-Zipで解凍 (フルパス指定でWinError 2を回避)
                    seven_zip_path = r"C:\Program Files\7-Zip\7z.exe"
                    
                    # もし上記でダメなら、こっちのパス↓の#を消して切り替えてみてください
                    # seven_zip_path = r"C:\Program Files (x86)\7-Zip\7z.exe"
                    
                    result = subprocess.run(
                        [seven_zip_path, 'e', lzh_save_path, f"-o{target['save_dir']}", '-y'],
                        capture_output=True, text=True
                    )

                    # 🗑️ 3. LZHファイルを削除
                    if os.path.exists(lzh_save_path):
                        os.remove(lzh_save_path)

                    # 🔍 4. 本当に解凍されたか最終確認
                    if os.path.exists(txt_save_path):
                        print(f"✅ 取得成功: {yyyy_mm}/{txt_name}")
                    else:
                        print(f"❌ 解凍失敗 ({file_name}): TXTファイルが生成されませんでした。")
                        if result.stderr:
                            print(f"   [7-Zipエラー詳細]: {result.stderr.strip()}")
                        elif result.stdout:
                            print(f"   [7-Zip出力ログ]: {result.stdout.strip()[:200]}...") # 長すぎるので200文字まで
                else:
                    print(f"⚠️ スキップ: {download_url} (ステータスコード: {response.status_code})")
            except Exception as e:
                print(f"⚠️ 予期せぬエラー発生 ({file_name}): {e}")

            time.sleep(0.5)

        current_date += timedelta(days=1)

    print("🏁 処理が終了しました。")

# 実行
download_and_extract_custom(START_DATE, END_DATE)