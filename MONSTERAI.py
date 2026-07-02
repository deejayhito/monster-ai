# ==========================================
# MONSTER AI - ローカル実行版 (D:\MONSTER AI 用)
# セル1(パース) + セル2(特徴量・学習・集計) + セル3(HTML生成) を1本化
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import unicodedata
import time
import json
import shutil
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 🌟 0. パス設定（ローカル用）
# ==========================================
BASE_DIR = r'D:\MONSTER AI'
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
BANGUMI_DIR = os.path.join(BASE_DIR, 'bangumihyou')
KYOUSOU_DIR = os.path.join(BASE_DIR, 'kyousouseiseki')
RACER_PATH = os.path.join(BASE_DIR, 'racerkibetsuseiseki', 'fan2604.txt')
PUBLISH_DIR = os.path.join(BASE_DIR, 'publish')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PUBLISH_DIR, exist_ok=True)

# 今日の日付を自動取得（YYMMDD形式）
target_date = datetime.now().strftime('%y%m%d')
print(f"=== 対象日付: {target_date} ===")

# ==========================================
# 🌟 セル1由来: 定数・パーサー関数
# ==========================================
MOTOR_UPDATE_DATES = {
    '桐生': '251227', '戸田': '250806', '江戸川': '260511', '平和島': '250609',
    '多摩川': '260418', '浜名湖': '260409', '蒲郡': '250719', '常滑': '251111',
    '津': '251222', '三国': '260307', 'びわこ': '260408', '住之江': '260323',
    '尼崎': '260417', '鳴門': '260411', '丸亀': '250903', '児島': '251217',
    '宮島': '251019', '徳山': '260420', '下関': '260429', '若松': '251126',
    '芦屋': '260416', '福岡': '260218', '唐津': '250905', '大村': '260524'
}

VENUE_HW_DEG = {
    '桐生': 157.5, '戸田': 112.5, '江戸川': 22.5, '平和島': 0, '多摩川': 270, '浜名湖': 180,
    '蒲郡': 247.5, '常滑': 270, '津': 292.5, '三国': 157.5, 'びわこ': 202.5, '住之江': 180,
    '尼崎': 247.5, '鳴門': 112.5, '丸亀': 135, '児島': 180, '宮島': 225, '徳山': 292.5,
    '下関': 225, '若松': 247.5, '芦屋': 90, '福岡': 67.5, '唐津': 202.5, '大村': 45
}

DIR_DEG = {
    '北':0, '北北東':22.5, '北東':45, '東北東':67.5, '東':90, '東南東':112.5, '南東':135, '南南東':157.5,
    '南':180, '南南西':202.5, '南西':225, '西南西':247.5, '西':270, '西北西':292.5, '北西':315, '北北西':337.5,
    '無風':-1, '無':-1
}

VENUES_SEARCH_LIST = ['唐津', '桐生', '戸田', '江戸川', '平和島', '多摩川', '浜名湖', '蒲郡', '常滑', '三国', 'びわこ', '住之江', '尼崎', '鳴門', '丸亀', '児島', '宮島', '徳山', '下関', '若松', '芦屋', '福岡', '大村', '津']

def get_relative_wind(venue, w_dir):
    w_dir_str = str(w_dir).strip()
    if w_dir_str not in DIR_DEG or DIR_DEG[w_dir_str] == -1: return "無風"
    w_deg = DIR_DEG[w_dir_str]
    hw_deg = VENUE_HW_DEG.get(venue, 0)
    diff = (w_deg - hw_deg + 360) % 360
    if diff > 180: diff -= 360
    if abs(diff) <= 45: return "向かい風"
    elif abs(diff) >= 135: return "追い風"
    elif diff > 0: return "右横風"
    else: return "左横風"

def clean_string(text):
    if not isinstance(text, str): return ""
    norm = unicodedata.normalize('NFKC', text)
    return re.sub(r'[\s\x00-\x1F\x7F-\x9F]+', '', norm)

def clean_day_num(day_str):
    day_str = clean_string(day_str)
    num_str = re.sub(r'[^\d]', '', day_str)
    if num_str: return int(num_str)
    m = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
    for k, v in m.items():
        if k in day_str: return v
    return 1

def get_grade_from_line(line, current_grade):
    u_text = clean_string(line).upper()
    sg_list = ["ボートレースクラシック", "ボートレースオールスター", "グランドチャンピオン", "オーシャンカップ", "ボートレースメモリアル", "ボートレースダービー", "チャレンジカップ", "グランプリシリーズ", "グランプリ"]
    is_sg = False
    has_round = bool(re.search(r'第[0-9〇一二三四五六七八九十百]+回', u_text))
    for sg in sg_list:
        if sg in u_text and has_round:
            is_sg = True
            break
    if is_sg: return "SG"

    if "レディースチャレンジカップ" in u_text or "レディースCC" in u_text: return "G2"
    if re.search(r'G2|GⅡ|GII|モーターボート大賞|MB大賞|モーターボート誕生祭|ボートレース甲子園|甲子園|レディースオールスター|秩父宮妃記念杯', u_text):
        return "G2"

    g1_list = [
        "赤城雷神杯", "戸田プリムローズ", "江戸川大賞", "トーキョー・ベイ・カップ", "トーキョーベイカップ",
        "ウェイキーカップ", "浜名湖賞", "オールジャパン竹島特別", "トコタンキング決定戦", "ツッキー王座決定戦",
        "北陸艇王決戦", "びわこ大賞", "太閤賞", "尼崎センプルカップ", "大渦大賞", "京極賞",
        "児島キングカップ", "宮島チャンピオンカップ", "徳山クラウン争奪戦", "競帝王決定戦", "全日本覇者決定戦",
        "全日本王座決定戦", "福岡チャンピオンカップ", "全日本王者決定戦", "海の王者決定戦",
        "地区選手権", "ダイヤモンドカップ", "高松宮記念", "ヤングダービー", "マスターズチャンピオン",
        "レディースチャンピオン", "クイーンズクライマックス", "BBCトーナメント", "スピードクイーンメモリアル", "名人戦"
    ]

    if any(g1 in u_text for g1 in g1_list):
        return "G1"
    elif re.search(r'G1|GⅠ', u_text):
        if not re.search(r'市制|町制|区制|村制|BTS|チケットショップ|ナイター', u_text):
            return "G1"

    g3_hit = any(g3 in u_text for g3 in [
        "オールレディース", "イースタンヤング", "ウエスタンヤング", "企業杯", "マスターズリーグ",
        "サッポロビールカップ", "キリンカップ", "アサヒビールカップ", "サントリーカップ"
    ])
    if g3_hit and "マスターズ" in u_text and "マスターズリーグ" not in u_text:
        g3_hit = False
    if g3_hit or re.search(r'G3|GⅢ|GIII', u_text):
        return "G3"

    return current_grade

def safe_float(val):
    try: return float(val)
    except ValueError: return 0.0

def safe_int(val):
    try: return int(val)
    except ValueError: return 0

def normalize_venue_name(name):
    clean = clean_string(name)
    for v in VENUES_SEARCH_LIST:
        if v in clean: return v
    fallback = {'びわ':'びわこ', '多摩':'多摩川', '下':'下関', '福':'福岡'}
    for k, v in fallback.items():
        if k in clean: return v
    return "不明"

def parse_bangumihyou_final(file_path):
    data = []
    filename = os.path.basename(file_path)
    date_str = filename[1:7] if filename.upper().startswith('B') else "000000"
    current_race, current_venue, current_race_round, current_deadline = None, "不明", "", ""
    current_grade = "一般"
    current_tournament_name = ""
    current_day_num = 1
    scanning_header = True
    found_title_mark = False
    row_pattern = re.compile(r'^\s*([1-6])\s+(\d{4}).*?([AB][12])\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)\s+(\d{1,3}\.\d{2})\s*(\d+)\s+([\d\.]+)')

    with open(file_path, 'r', encoding='cp932', errors='replace') as f:
        for line in f:
            line_normalized = unicodedata.normalize('NFKC', line)
            line_clean = clean_string(line_normalized)
            new_venue = "不明"
            if "ボートレース" in line_clean or "ボートレース" in line_normalized:
                for v in VENUES_SEARCH_LIST:
                    if v in line_clean:
                        new_venue = v; break

            if new_venue != "不明":
                if new_venue != current_venue or not scanning_header:
                    current_venue = new_venue
                    current_grade = "一般"
                    current_tournament_name = ""
                    scanning_header = True
                    found_title_mark = False

            if scanning_header:
                temp_grade = get_grade_from_line(line_normalized, current_grade)
                if "ルーキーシリーズ" in line_clean or "ヴィーナスシリーズ" in line_clean: current_grade = "一般"
                elif temp_grade != "一般": current_grade = temp_grade

                day_match = re.search(r'第\s*([0-9一二三四五六七八九十０-９最終準優勝]+)\s*日', line_normalized)
                if day_match: current_day_num = clean_day_num(day_match.group(1))
                elif "最終日" in line_normalized: current_day_num = "最終日"
                elif "優勝戦日" in line_normalized: current_day_num = "優勝戦日"

                if "番組表" in line_normalized: found_title_mark = True; continue
                if found_title_mark and line_clean:
                    if "主催者発行" in line_clean: continue
                    if re.search(r'\d{4}年', line_clean) or re.search(r'^第\d+日', line_clean): continue
                    current_tournament_name = line_normalized.replace(' ', '').strip()
                    found_title_mark = False

            race_match = re.search(r'^\s*(\d+)R', line_normalized, re.IGNORECASE)
            if race_match:
                scanning_header = False
                current_race = int(race_match.group(1))
                name_match = re.search(r'\d+R\s+(.*?)\s+H\d{4}m', line_normalized, re.IGNORECASE)
                current_race_round = name_match.group(1).strip() if name_match else ""
                time_match = re.search(r'(\d{1,2}:\d{2})', line_normalized)
                current_deadline = time_match.group(1) if time_match else ""
                continue

            match = row_pattern.search(line_normalized)
            if current_race and match:
                has_f = 1 if 'F' in line_normalized.upper() else 0
                data.append({
                    'Date': date_str, '場': current_venue, 'グレード': current_grade, 'R': current_race,
                    'レース名': current_tournament_name, 'レース種別': current_race_round, '締切時間': current_deadline,
                    '艇番': safe_int(match.group(1)), '登録番号': str(match.group(2)), '級別': match.group(3),
                    '全国勝率': safe_float(match.group(4)), '全国2連率': safe_float(match.group(5)),
                    '当地勝率': safe_float(match.group(6)), '当地2連率': safe_float(match.group(7)),
                    'モータNO': safe_int(match.group(8)), 'モータ2連率': safe_float(match.group(9)),
                    'ボートNO': safe_int(match.group(10)), 'ボート2連率': safe_float(match.group(11)), 'F持ち': has_f,
                    'DayNum': current_day_num
                })
    return pd.DataFrame(data)

def parse_kyousouseiseki_incremental(folder_path, existing_dates):
    records = []
    race_meta = {}
    if not os.path.exists(folder_path): return pd.DataFrame()
    race_pattern = re.compile(r'^\s*(\d+)R\s+')
    row_pattern = re.compile(r'^\s*([0-6A-Z]{1,2})\s+([1-6])\s+(\d{4})')
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.upper().endswith('.TXT') and file.upper().startswith('K'):
                if file[1:7] not in existing_dates: files_to_process.append(os.path.join(root, file))

    for file_path in tqdm(files_to_process, desc="成績ロード中"):
        date_str = os.path.basename(file_path)[1:7]
        current_race, current_venue = None, "不明"
        current_day_num = 1
        scanning_header = True
        current_payout_type = None

        with open(file_path, 'r', encoding='cp932', errors='replace') as f:
            for line in f:
                line_norm = unicodedata.normalize('NFKC', line)
                line_clean = clean_string(line_norm)
                new_venue = "不明"
                if "［成績］" in line_norm or "[成績]" in line_norm or "成績" in line_clean:
                    line_alt = line_norm.replace('［', '[').replace('］', ']')
                    new_venue = normalize_venue_name(line_alt.split("[")[0].strip() if "[" in line_alt else line_alt)
                elif "ボートレース" in line_clean or "ボートレース" in line_norm:
                    for v in VENUES_SEARCH_LIST:
                        if v in line_clean: new_venue = v; break

                if new_venue != "不明":
                    if new_venue != current_venue or not scanning_header:
                        current_venue = new_venue
                        scanning_header = True

                if scanning_header:
                    day_match = re.search(r'第\s*([0-9一二三四五六七八九十０-９最終準優勝]+)\s*日', line_norm)
                    if day_match: current_day_num = clean_day_num(day_match.group(1))
                    elif "最終日" in line_norm: current_day_num = "最終日"
                    elif "優勝戦日" in line_norm: current_day_num = "優勝戦日"

                r_match = race_pattern.search(line_norm)
                if r_match:
                    scanning_header = False
                    current_race = int(r_match.group(1))
                    current_payout_type = None
                    key = (date_str, current_venue, current_race)
                    if key not in race_meta:
                        race_meta[key] = {'決まり手': '－', '天候': '晴れ', '風向': '無風', '相対風向': '無風', '風速': 0, '波高': 0, '複勝': [], '拡連複': []}
                    w_match = re.search(r'H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm', line_norm)
                    if w_match:
                        race_meta[key]['天候'] = w_match.group(1).strip()
                        w_dir = w_match.group(2).strip()
                        race_meta[key]['風向'] = w_dir
                        race_meta[key]['相対風向'] = get_relative_wind(current_venue, w_dir)
                        race_meta[key]['風速'] = safe_float(w_match.group(3))
                        race_meta[key]['波高'] = safe_float(w_match.group(4))
                    continue

                key = (date_str, current_venue, current_race)
                if current_race and current_venue != "不明":
                    if "レースタイム" in line_norm:
                        kima_str = line_norm.split("レースタイム")[-1].replace(' ', '').replace(' ', '').strip()
                        for vk in ['逃げ', '差し', 'まくり差し', 'まくり', '抜き', '恵まれ']:
                            if vk in kima_str:
                                if key in race_meta: race_meta[key]['決まり手'] = vk
                                break
                    if "単勝" in line_norm: current_payout_type = "単勝"
                    elif "複勝" in line_norm: current_payout_type = "複勝"
                    elif "２連単" in line_norm or "2連単" in line_norm: current_payout_type = "二連単"
                    elif "２連複" in line_norm or "2連複" in line_norm: current_payout_type = "二連複"
                    elif "拡連複" in line_norm: current_payout_type = "拡連複"
                    elif "３連単" in line_norm or "3連単" in line_norm: current_payout_type = "三連単"
                    elif "３連複" in line_norm or "3連複" in line_norm: current_payout_type = "三連複"

                    if current_payout_type == "複勝" and "複勝" in line_norm:
                        m = re.findall(r'(\d)\s+([\d,]+)', line_norm.split("複勝")[1])
                        if m and key in race_meta: race_meta[key]['複勝'] = m
                    elif current_payout_type == "単勝" and "単勝" in line_norm:
                        m = re.search(r'単勝\s+(\S+)\s+([\d,]+)', line_norm)
                        if m and key in race_meta:
                            race_meta[key]['単勝_combo'] = m.group(1)
                            race_meta[key]['単勝_payout'] = m.group(2)
                    else:
                        m = re.search(r'(\d-\d(?:-\d)?)\s+([\d,]+)\s+人気\s+(\d+)', line_norm)
                        if m and key in race_meta and current_payout_type:
                            combo, payout, pop = m.groups()
                            if current_payout_type == "拡連複": race_meta[key]['拡連複'].append((combo, payout, pop))
                            else:
                                race_meta[key][current_payout_type + '_combo'] = combo
                                race_meta[key][current_payout_type + '_payout'] = payout
                                race_meta[key][current_payout_type + '_pop'] = pop

                row_match = row_pattern.search(line_norm)
                if current_race and row_match:
                    rank = row_match.group(1).lstrip('0')
                    sub_line = line_norm[row_match.end():]
                    tenji, guessing_course, st_val, st_disp = np.nan, int(row_match.group(2)), np.nan, "-"
                    tenji_match = re.search(r'\s(6\.\d{2}|7\.\d{2})\s', sub_line)
                    if tenji_match: tenji = safe_float(tenji_match.group(1))
                    matches = re.findall(r'\s([1-6])\s+([FL]?\d{0,2}\.\d{2})', sub_line)
                    if matches:
                        guessing_course = int(matches[-1][0])
                        st_str = matches[-1][1]
                        st_disp = st_str.replace('0.', '.')
                        if 'F' not in st_str and 'L' not in st_str:
                            clean_st = '0' + st_str if st_str.startswith('.') else st_str
                            st_val = safe_float(clean_st)

                    parts = line_norm.strip().split()
                    race_time = parts[-1] if len(parts) > 0 else "-"
                    records.append({
                        'Date': date_str, '場': current_venue, 'R': current_race,
                        '艇番': safe_int(row_match.group(2)), '登録番号': str(row_match.group(3)),
                        'コース': guessing_course, '着順': rank, 'ST': st_val, 'ST_表示': st_disp,
                        '展示': tenji, 'DayNum': current_day_num, 'レースタイム': race_time
                    })

    df_records = pd.DataFrame(records)
    if not df_records.empty and race_meta:
        meta_list = []
        for k, v in race_meta.items():
            v['Date'] = k[0]; v['場'] = k[1]; v['R'] = k[2]
            meta_list.append(v)
        df_records = pd.merge(df_records, pd.DataFrame(meta_list), on=['Date', '場', 'R'], how='left')
    return df_records

def parse_racer_data(file_path):
    records = []
    if not os.path.exists(file_path): return pd.DataFrame()
    with open(file_path, 'rb') as f:
        for line in f:
            if len(line) < 82: continue
            try:
                toban = line[0:4].decode('cp932').strip()
                raw_name = line[4:20].decode('cp932', errors='ignore')
                name_fmt = re.sub(r'[  ]{2,}', '★', raw_name)
                name_fmt = re.sub(r'[  ]', '', name_fmt)
                name = name_fmt.replace('★', ' ').strip()
                branch = line[35:39].decode('cp932', errors='ignore').replace(' ', '').replace(' ', '').strip()
                dob_raw_line = line.decode('cp932', errors='ignore')
                dob_match = re.search(r'[SHTR]\d{6}', dob_raw_line)
                dob = f"{dob_match.group(0)[:3]}/{dob_match.group(0)[3:5]}/{dob_match.group(0)[5:]}" if dob_match else "-"
                gender_code = line[48:49].decode('cp932').strip()
                is_female = 1 if gender_code == '2' else 0
                age = int(line[49:51].decode('cp932').strip())
                height = line[51:54].decode('cp932').strip()
                weight = int(line[54:56].decode('cp932').strip())
                avg_st = safe_float(line[79:82].decode('cp932').strip()) / 100
                records.append({'登録番号': toban, '選手名': name, '支部': branch, '生年月日': dob, 'is_female': is_female, '年齢': age, '身長': height, '体重': weight, '平均ST': avg_st})
            except: continue
    return pd.DataFrame(records)

# ==========================================
# 🌟 セル1由来: メイン実行部分（データ読み込み）
# ==========================================
start_time = time.time()

seiseki_cache_path = os.path.join(CACHE_DIR, 'df_seiseki_3years_v24.pkl')
bangumi_train_cache_path = os.path.join(CACHE_DIR, 'df_train_3years_v24.pkl')

print("--- 1. 競走成績(Kファイル)のロード ---")
if os.path.exists(seiseki_cache_path):
    print("✅ キャッシュファイルを発見しました。読み込み中...")
    df_seiseki = pd.read_pickle(seiseki_cache_path)
    existing_seiseki_dates = set(df_seiseki['Date'].unique())
else:
    print("⏳ 完全取得版キャッシュを作成中...")
    df_seiseki = pd.DataFrame()
    existing_seiseki_dates = set()

df_new_seiseki = parse_kyousouseiseki_incremental(KYOUSOU_DIR, existing_seiseki_dates)
if not df_new_seiseki.empty:
    df_seiseki = pd.concat([df_seiseki, df_new_seiseki], ignore_index=True).drop_duplicates(subset=['Date', '場', 'R', '艇番']).reset_index(drop=True)
    df_seiseki.to_pickle(seiseki_cache_path)

racer_df = parse_racer_data(RACER_PATH)

print("\n--- 2. 番組表(Bファイル)のロード ---")
test_data_list = []
train_data_list = []
files_to_process_b = []

if os.path.exists(bangumi_train_cache_path):
    df_train = pd.read_pickle(bangumi_train_cache_path)
    existing_train_dates = set(df_train['Date'].unique())
else:
    df_train = pd.DataFrame()
    existing_train_dates = set()

for root, _, files in os.walk(BANGUMI_DIR):
    for file in files:
        if file.upper().startswith('B') and file.upper().endswith('.TXT'):
            date_str = file[1:7]
            if date_str == target_date or (date_str < target_date and date_str not in existing_train_dates):
                files_to_process_b.append(os.path.join(root, file))

for file_path in tqdm(files_to_process_b, desc="番組表ロード中"):
    date_str = os.path.basename(file_path)[1:7]
    df_b = parse_bangumihyou_final(file_path)
    if not df_b.empty:
        if date_str == target_date: test_data_list.append(df_b)
        else: train_data_list.append(df_b)

df_test = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()

if train_data_list:
    df_train = pd.concat([df_train] + train_data_list, ignore_index=True).drop_duplicates(subset=['Date', '場', 'R', '艇番']).reset_index(drop=True)
    df_train.to_pickle(bangumi_train_cache_path)

all_b_df = pd.concat([df_train, df_test]) if not df_train.empty else df_test

if all_b_df.empty:
    print("\n⚠️ エラー：読み込める番組表(Bファイル)が0件です！")
else:
    if 'Date' in all_b_df.columns and '場' in all_b_df.columns:
        grade_map = all_b_df.groupby(['Date', '場'])['グレード'].first().to_dict()
        df_seiseki['グレード'] = pd.Series(list(zip(df_seiseki['Date'], df_seiseki['場'])), index=df_seiseki.index).map(grade_map).fillna(df_seiseki.get('グレード', '一般'))
        type_map = all_b_df.groupby(['Date', '場', 'R'])['レース種別'].first().to_dict()
        df_seiseki['レース種別'] = pd.Series(list(zip(df_seiseki['Date'], df_seiseki['場'], df_seiseki['R'])), index=df_seiseki.index).map(type_map).fillna('一般')
    print("✅ 【パート1】完了しました！")

# ==========================================
# 🌟 重複エラー＆型ズレ完全防止ガード ＋ 必須関数
# ==========================================
cleanup_cols = [
    '着順', 'ST_表示', '展示', '風速', '波高', '決まり手', '天候', '風向', '相対風向',
    '単勝_combo', '単勝_payout', '二連単_combo', '二連単_payout', '二連単_pop',
    '二連複_combo', '二連複_payout', '二連複_pop', '三連単_combo', '三連単_payout',
    '三連単_pop', '三連複_combo', '三連複_payout', '三連複_pop', 'レースタイム', '複勝', '拡連複',
    '当日展示タイム', '当日風速', '当日波高', '展示タイム差', '相対風向_num', '1着艇番',
    '同枠勝率', '同枠3連対率', '枠別10走_平均着順', '枠別10走_1着回数', '同枠過去10走_平均ST',
    '過去_逃げ回数', '過去_まくり回数', '過去_差し回数', '過去_まくり差し回数', 'F回数', 'L回数', 'S回数',
    'モータ2連率_raw', 'F持ち_平均ST_積', 'F持ち_当地2連率_商'
]
for c in ['複勝', '拡連複']:
    if c not in df_seiseki.columns: df_seiseki[c] = None
df_train = df_train.loc[:, ~df_train.columns.duplicated()].copy()
df_train.drop(columns=[c for c in df_train.columns if c in cleanup_cols or c.endswith('_drop')], inplace=True, errors='ignore')
df_test = df_test.loc[:, ~df_test.columns.duplicated()].copy()
df_test.drop(columns=[c for c in df_test.columns if c in cleanup_cols or c.endswith('_drop')], inplace=True, errors='ignore')
df_train['登録番号'] = df_train['登録番号'].astype(str)
df_test['登録番号'] = df_test['登録番号'].astype(str)
df_seiseki['登録番号'] = df_seiseki['登録番号'].astype(str)
def safe_rank_convert(x):
    return int(str(x).strip()) if str(x).strip() in ['1', '2', '3', '4', '5', '6'] else 7
def get_clean_rank(rk):
    s = str(rk).strip()
    if s and s[0].isalpha(): return s[0]
    return s
def get_maru_rank(rk, race_type, r_num, day_num):
    s = str(rk).strip()
    is_yusho = False
    rtype = str(race_type)
    if '優勝戦' in rtype and '準' not in rtype: is_yusho = True
    elif str(day_num) in ['最終日', '優勝戦日'] and int(r_num) == 12: is_yusho = True

    if is_yusho and s in ['1', '2', '3', '4', '5', '6']:
        return {'1':'①','2':'②','3':'③','4':'④','5':'⑤','6':'⑥'}[s]
    if s and s[0].isalpha(): return s[0]
    return s
def safe_list(val):
    if isinstance(val, list): return val
    return []
def format_racer_name(raw_name):
    if not isinstance(raw_name, str): return str(raw_name)
    name_no_space = re.sub(r'[  ]', '', raw_name)
    if len(name_no_space) >= 4:
        return name_no_space[:2] + ' ' + name_no_space[2:]
    return name_no_space
def get_unique_pos(pos_list, exclude_vals):
    if not isinstance(exclude_vals, list): exclude_vals = [exclude_vals]
    return sorted(list(set([p for p in pos_list if p not in exclude_vals])))
def calc_pts(pos1, pos2, pos3):
    pts = 0
    for p1 in pos1:
        for p2 in pos2:
            if p1 == p2: continue
            for p3 in pos3:
                if p1 == p3 or p2 == p3: continue
                pts += 1
    return pts
# ==========================================
# 🌟 1. データ前処理 ＆ 特徴量計算（超高速化版）
# ==========================================
print("\n--- 3. データ前処理 ＆ 特徴量計算中 ---")
df_s = df_seiseki.copy().sort_values(['Date', 'R'])
df_s['is_win'] = (df_s['着順'] == '1').astype(int)
df_s['is_top3'] = (df_s['着順'].isin(['1', '2', '3'])).astype(int)
df_s['num_rank'] = df_s['着順'].apply(safe_rank_convert)
df_s['is_nige'] = ((df_s['着順'] == '1') & (df_s['決まり手'] == '逃げ')).astype(int)
df_s['is_makuri'] = ((df_s['着順'] == '1') & (df_s['決まり手'] == 'まくり')).astype(int)
df_s['is_sashi'] = ((df_s['着順'] == '1') & (df_s['決まり手'] == '差し')).astype(int)
df_s['is_makurisashi'] = ((df_s['着順'] == '1') & (df_s['決まり手'] == 'まくり差し')).astype(int)
col_map = {'過去_逃げ回数': 'is_nige', '過去_まくり回数': 'is_makuri', '過去_差し回数': 'is_sashi', '過去_まくり差し回数': 'is_makurisashi'}
for col, base in tqdm(col_map.items(), desc="特徴量生成(決まり手)"):
    df_s[col] = df_s.groupby('登録番号')[base].transform(lambda x: x.shift(1).rolling(50, min_periods=1).sum()).fillna(0)
print("⏳ 枠番ごとの統計特徴量（勝率・平均ST・着順など）を計算中...")
results = []
for keys, g in tqdm(df_s.groupby(['登録番号', '艇番']), desc="特徴量生成(選手×枠番別統計)"):
    g = g.sort_values('Date')
    valid_st = g['ST'].dropna()
    if not valid_st.empty: g['同枠過去10走_平均ST'] = valid_st.shift(1).rolling(10, min_periods=1).mean().reindex(g.index).ffill()
    else: g['同枠過去10走_平均ST'] = np.nan

    g['同枠勝率'] = g['is_win'].shift(1).expanding().mean().fillna(0)
    g['同枠3連対率'] = g['is_top3'].shift(1).expanding().mean().fillna(0)
    g['枠別10走_平均着順'] = g['num_rank'].shift(1).rolling(10, min_periods=1).mean().fillna(3.5)
    g['枠別10走_1着回数'] = g['is_win'].shift(1).rolling(10, min_periods=1).sum().fillna(0)
    results.append(g)
df_s = pd.concat(results).sort_index()
merge_cols_s = ['Date', 'R', '登録番号', '艇番', '同枠勝率', '同枠3連対率', '枠別10走_平均着順', '枠別10走_1着回数', '同枠過去10走_平均ST', '過去_逃げ回数', '過去_まくり回数', '過去_差し回数', '過去_まくり差し回数']
if not df_train.empty: df_train = pd.merge(df_train, df_s[merge_cols_s], on=['Date', 'R', '登録番号', '艇番'], how='left')
df_s_past = df_s[df_s['Date'] < target_date].copy()
latest_stats = df_s_past.groupby(['登録番号', '艇番']).agg(
    同枠勝率=('is_win', 'mean'), 同枠3連対率=('is_top3', 'mean'),
    枠別10走_平均着順=('num_rank', lambda x: x.tail(10).mean()), 枠別10走_1着回数=('is_win', lambda x: x.tail(10).sum()),
    過去_逃げ回数=('is_nige', lambda x: x.tail(50).sum()), 過去_まくり回数=('is_makuri', lambda x: x.tail(50).sum()),
    過去_差し回数=('is_sashi', lambda x: x.tail(50).sum()), 過去_まくり差し回数=('is_makurisashi', lambda x: x.tail(50).sum())
).reset_index()
latest_st_df = df_s_past.groupby(['登録番号', '艇番'])['ST'].agg(lambda x: x.dropna().tail(10).mean() if not x.dropna().empty else np.nan).reset_index().rename(columns={'ST': '同枠過去10走_平均ST'})
latest_stats = pd.merge(latest_stats, latest_st_df, on=['登録番号', '艇番'], how='left')
if not df_test.empty: df_test = pd.merge(df_test, latest_stats, on=['登録番号', '艇番'], how='left')
# ==========================================
# 🌟 2. 環境特徴量 ＆ 事故ペナルティ集計
# ==========================================
def engineer_basic_features(df_base):
    if df_base.empty: return df_base
    df_base['グレード_num'] = df_base['グレード'].map({'SG': 4, 'G1': 3, 'G2': 2, 'G3': 1, '一般': 0}).fillna(0)
    df_base['級別_num'] = df_base['級別'].map({'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}).fillna(0)
    df_base['is_womens_race'] = df_base['レース名'].astype(str).apply(lambda x: 1 if any(k in x for k in ['レディース', 'ヴィーナス', '女子', 'W優勝']) else 0)

    if not racer_df.empty:
        cols = racer_df.columns.difference(df_base.columns).tolist() + ['登録番号']
        df_base = pd.merge(df_base, racer_df[cols], on='登録番号', how='left')

    if 'is_female' in df_base.columns: df_base['is_female'] = df_base['is_female'].fillna(0).astype(int)
    else: df_base['is_female'] = 0
    df_base = df_base.sort_values(['Date', '場', 'R', '艇番'])
    df_base['内隣ST'] = df_base.groupby(['Date', '場', 'R'])['平均ST'].shift(1).fillna(df_base['平均ST'])
    df_base['外隣ST'] = df_base.groupby(['Date', '場', 'R'])['平均ST'].shift(-1).fillna(df_base['平均ST'])
    df_base['内枠ST差'] = (df_base['平均ST'] - df_base['内隣ST']).fillna(0)
    df_base['外枠ST差'] = (df_base['外隣ST'] - df_base['平均ST']).fillna(0)
    df_base['F持ち_平均ST_積'] = df_base['F持ち'] * df_base['平均ST']
    df_base['F持ち_当地2連率_商'] = df_base['当地2連率'] / (df_base['F持ち'] + 1.0)
    return df_base
df_train = engineer_basic_features(df_train)
df_test = engineer_basic_features(df_test)
target_seiseki_cols = ['Date', 'R', '登録番号', '着順', 'ST_表示', '展示', '風速', '波高', '決まり手', '天候', '風向', '相対風向', '単勝_combo', '単勝_payout', '二連単_combo', '二連単_payout', '二連単_pop', '二連複_combo', '二連複_payout', '二連複_pop', '三連単_combo', '三連単_payout', '三連単_pop', '三連複_combo', '三連複_payout', '三連複_pop', 'レースタイム', '複勝', '拡連複']
if not df_train.empty:
    df_train = pd.merge(df_train, df_seiseki[target_seiseki_cols], on=['Date', 'R', '登録番号'], how='left')
    df_train = df_train.dropna(subset=['着順'])
    df_train['着順'] = df_train['着順'].astype(str)
if not df_test.empty:
    df_test = pd.merge(df_test, df_seiseki[target_seiseki_cols].drop(columns=['着順']), on=['Date', 'R', '登録番号'], how='left')
wind_map = {"無風": 0, "向かい風": 1, "追い風": 2, "左横風": 3, "右横風": 4}
for df in [df_train, df_test]:
    if df.empty: continue
    df.rename(columns={'展示': '当日展示タイム', '風速': '当日風速', '波高': '当日波高'}, inplace=True)
    df['当日風速'] = pd.to_numeric(df['当日風速'], errors='coerce').fillna(0)
    df['当日波高'] = pd.to_numeric(df['当日波高'], errors='coerce').fillna(0)
    df['当日展示タイム'] = pd.to_numeric(df['当日展示タイム'], errors='coerce')

    mean_tenji = df.groupby(['Date', '場', 'R'])['当日展示タイム'].transform('mean')
    df['当日展示タイム'] = df['当日展示タイム'].fillna(mean_tenji).fillna(6.80)
    df['展示タイム差'] = (df['当日展示タイム'] - mean_tenji).fillna(0)

    if '相対風向' not in df.columns: df['相対風向'] = '無風'
    df['相対風向'] = df['相対風向'].fillna('無風')
    df['相対風向_num'] = df['相対風向'].map(wind_map).fillna(0).astype(int)
def get_period_id(date_str):
    y, m = int(date_str[0:2]), int(date_str[2:4])
    if 5 <= m <= 10: return f"{y:02d}S"
    elif m >= 11: return f"{y:02d}A"
    else: return f"{y-1:02d}A"
def attach_penalties(df_target, df_seiseki):
    if df_seiseki.empty or df_target.empty:
        df_target['F回数'] = 0; df_target['L回数'] = 0; df_target['S回数'] = 0
        return df_target
    temp = df_seiseki[['Date', '登録番号', '着順']].copy()
    temp['期'] = temp['Date'].apply(get_period_id)
    temp['is_F'] = (temp['着順'] == 'F').astype(int)
    temp['is_L'] = (temp['着順'] == 'L').astype(int)
    temp['is_S'] = temp['着順'].apply(lambda x: 1 if str(x).startswith('S') else 0).astype(int)
    unique_dates = df_target['Date'].unique()
    penalty_records = []

    for d in tqdm(unique_dates, desc="特徴量生成(事故ペナルティ集計)"):
        period = get_period_id(d)
        mask = (temp['期'] == period) & (temp['Date'] < d)
        hist = temp[mask].groupby('登録番号')[['is_F', 'is_L', 'is_S']].sum().reset_index()
        hist['Date'] = d
        penalty_records.append(hist)

    if penalty_records:
        hist_df = pd.concat(penalty_records)
        df_target = pd.merge(df_target, hist_df, on=['Date', '登録番号'], how='left')
        df_target = df_target.rename(columns={'is_F': 'F回数', 'is_L': 'L回数', 'is_S': 'S回数'})
    else:
        df_target['F回数'] = 0; df_target['L回数'] = 0; df_target['S回数'] = 0
    df_target[['F回数', 'L回数', 'S回数']] = df_target[['F回数', 'L回数', 'S回数']].fillna(0).astype(int)
    return df_target
if '同枠過去10走_平均ST' in df_train.columns: df_train['同枠過去10走_平均ST'] = df_train['同枠過去10走_平均ST'].fillna(df_train['平均ST']).fillna(0.17)
if '同枠過去10走_平均ST' in df_test.columns: df_test['同枠過去10走_平均ST'] = df_test['同枠過去10走_平均ST'].fillna(df_test['平均ST']).fillna(0.17)
df_train, df_test = attach_penalties(df_train, df_seiseki), attach_penalties(df_test, df_seiseki)
for df in [df_train, df_test]:
    if df.empty: continue
    df['モータ2連率_raw'] = df['モータ2連率']
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%y%m%d')
    for venue, update_date in MOTOR_UPDATE_DATES.items():
        udate = pd.to_datetime(update_date, format='%y%m%d')
        mask = (df['場'] == venue) & (df['Date_dt'] >= udate) & (df['Date_dt'] <= udate + pd.Timedelta(days=45))
        df.loc[mask, 'モータ2連率'] = 33.0
    df.drop(columns=['Date_dt'], inplace=True)
# ==========================================
# 🌟 3. AIエンジン 学習 ＆ 確率予測
# ==========================================
feature_cols = [
    '艇番', 'グレード_num', '級別_num', 'is_female', 'is_womens_race',
    '全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モータNO', 'モータ2連率', 'ボートNO', 'ボート2連率',
    '年齢', '体重', '平均ST', 'F回数', 'L回数', 'S回数', 'F持ち',
    '同枠勝率', '同枠3連対率', '内枠ST差', '外枠ST差',
    '枠別10走_1着回数', '枠別10走_平均着順', '同枠過去10走_平均ST',
    '当日展示タイム', '展示タイム差', '当日風速', '当日波高',
    '過去_逃げ回数', '過去_まくり回数', '過去_差し回数', '過去_まくり差し回数',
    '相対風向_num', 'F持ち_平均ST_積', 'F持ち_当地2連率_商'
]
feature_cols = [c for c in feature_cols if c in df_train.columns]
print("\n--- AIエンジン 学習中 ---")
model_1 = lgb.train({'objective':'binary', 'learning_rate':0.05, 'verbose':-1}, lgb.Dataset(df_train[feature_cols].fillna(0), label=(df_train['着順'] == '1').astype(int), categorical_feature=['相対風向_num'] if '相対風向_num' in feature_cols else []), num_boost_round=150)
winner_df = df_train[df_train['着順'] == '1'][['Date', '場', 'R', '艇番']].rename(columns={'艇番': '1着艇番'}).drop_duplicates(subset=['Date', '場', 'R'])
df_train = pd.merge(df_train, winner_df, on=['Date', '場', 'R'], how='left').fillna({'1着艇番': 1})
df_train['1着艇番'] = df_train['1着艇番'].astype(int)
model_3 = lgb.train({'objective':'binary', 'learning_rate':0.05, 'verbose':-1}, lgb.Dataset(df_train[feature_cols + ['1着艇番']].fillna(0), label=(df_train['着順'].isin(['1', '2', '3'])).astype(int), categorical_feature=['1着艇番', '相対風向_num'] if '相対風向_num' in feature_cols else ['1着艇番']), num_boost_round=150)
print("\n--- 全日程のAI予測確率を計算中 ---")
combined_b_df = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=['Date', '場', 'R', '艇番']).reset_index(drop=True)
combined_b_df['prob_1_raw'] = model_1.predict(combined_b_df[feature_cols].fillna(0))
for i in tqdm(range(1, 7), desc="AI予測(3連対率)"):
    temp_c = combined_b_df[feature_cols].copy()
    temp_c['1着艇番'] = i
    combined_b_df[f'prob_3_given_{i}'] = model_3.predict(temp_c[feature_cols + ['1着艇番']].fillna(0))
b_motor_lookup = combined_b_df.drop_duplicates(subset=['Date', '場', 'R', '艇番']).set_index(['Date', '場', 'R', '艇番'])
# ==========================================
# 🌟 4. モーター履歴＆実績の集計
# ==========================================
print("\n--- モーター履歴＆実績を集計中 ---")
df_seiseki['Date'], df_seiseki['場'] = df_seiseki['Date'].astype(str), df_seiseki['場'].astype(str)
df_seiseki['R'], df_seiseki['艇番'] = df_seiseki['R'].astype(int), df_seiseki['艇番'].astype(int)
df_s_motor = df_seiseki.join(b_motor_lookup[['モータNO', '選手名', '級別', 'is_female']], on=['Date', '場', 'R', '艇番'], how='inner')
motor_history_dict, motor_stats_dict, venue_motor_lists = {}, {}, {}
latest_motors = df_test if not df_test.empty else combined_b_df
for venue, v_group in tqdm(combined_b_df.groupby('場'), desc="モーター集計"):
    update_date_str = MOTOR_UPDATE_DATES.get(venue, '000000')
    v_motor_df = df_s_motor[(df_s_motor['場'] == venue) & (df_s_motor['Date'] >= update_date_str) & (df_s_motor['Date'] < target_date)]
    motor_history_dict[venue], motor_stats_dict[venue] = {}, {}
    v_latest = latest_motors[latest_motors['場'] == venue]

    if not v_latest.empty:
        motor_ren2_map = v_latest.groupby('モータNO')['モータ2連率_raw'].max().to_dict()
        sorted_motors = sorted(motor_ren2_map.items(), key=lambda x: x[1], reverse=True)
        motor_rank_map = {m: i+1 for i, (m, _) in enumerate(sorted_motors)}
    else: motor_rank_map, motor_ren2_map = {}, {}

    for motor_no, m_group in v_motor_df.groupby('モータNO'):
        m_group = m_group.sort_values(['Date', 'R'])
        m_group['Date_dt'] = pd.to_datetime(m_group['Date'], format='%y%m%d')
        m_group['days_diff'] = (m_group['Date_dt'] - m_group['Date_dt'].shift(1)).dt.days
        m_group['user_diff'] = (m_group['登録番号'] != m_group['登録番号'].shift(1))
        m_group['setsu_id'] = ((m_group['days_diff'] > 4) | m_group['user_diff']).cumsum()

        history_list = []
        yushu_count, yusho_count = 0, 0
        for s_id, s_group in m_group.groupby('setsu_id'):
            racer_name = str(s_group['選手名'].iloc[0]) if pd.notna(s_group['選手名'].iloc[0]) else "不明"
            kyu = str(s_group['級別'].iloc[0]) if pd.notna(s_group['級別'].iloc[0]) else "B2"
            is_f = bool('is_female' in s_group.columns and s_group['is_female'].iloc[0] == 1)
            d_max = s_group['Date_dt'].max()

            last_day_races = m_group[m_group['Date_dt'] == d_max]
            if 12 in last_day_races['R'].values:
                yushu_count += 1
                if get_clean_rank(last_day_races[last_day_races['R'] == 12].iloc[0]['着順']) == '1': yusho_count += 1

            r_ranks = []
            for _, r in s_group.iterrows():
                rk = get_clean_rank(r['着順'])
                if r['Date_dt'] == d_max and r['R'] == 12 and rk in ['1','2','3','4','5','6']: rk = {'1':'①','2':'②','3':'③','4':'④','5':'⑤','6':'⑥'}[rk]
                r_ranks.append(rk)

            history_list.append({'date_str': f"{s_group['Date_dt'].min().month}/{s_group['Date_dt'].min().day}-{d_max.month}/{d_max.day}", 'racer_name': racer_name, 'kyu': kyu, 'ranks': "".join(r_ranks), 'date_max': d_max, 'is_female': is_f})

        history_list.sort(key=lambda x: x['date_max'], reverse=True)
        motor_history_dict[venue][motor_no] = [{'date_str': h['date_str'], 'racer_name': h['racer_name'], 'kyu': h['kyu'], 'ranks': h['ranks'], 'is_female': h['is_female']} for h in history_list]
        motor_stats_dict[venue][motor_no] = {'yushu': yushu_count, 'yusho': yusho_count, 'rank': motor_rank_map.get(motor_no, "-")}

    v_motor_list = []
    for m_no, ren2 in motor_ren2_map.items():
        st = motor_stats_dict[venue].get(m_no, {'yushu': 0, 'yusho': 0, 'rank': '-'})
        v_motor_list.append({'motor_no': m_no, 'motor_2ren': f"{float(ren2):.1f}", 'motor_rank': st['rank'], 'motor_yushu': st['yushu'], 'motor_yusho': st['yusho'], 'motor_history': motor_history_dict[venue].get(m_no, [])})
    venue_motor_lists[venue] = sorted(v_motor_list, key=lambda x: x['motor_rank'] if isinstance(x['motor_rank'], int) else 999)
# ==========================================
# 🌟 5. 選手詳細成績の事前集計
# ==========================================
print("\n--- 選手ダッシュボード用 詳細成績を構築中 ---")
df_seiseki_past = df_seiseki[df_seiseki['Date'] < target_date].sort_values('Date')
z10_dict = {}
for (toban, teiban), grp in tqdm(df_seiseki_past.groupby(['登録番号', '艇番']), desc="全国過去10走"):
    z10_dict[(str(toban), int(teiban))] = [{
        'venue': str(r['場'])[:1],
        'course': int(r.get('コース', teiban)) if pd.notna(r.get('コース')) else teiban,
        'st': str(r['ST_表示']),
        'rank': get_maru_rank(r['着順'], r.get('レース種別', '一般'), r['R'], r.get('DayNum', ''))
    } for _, r in grp.tail(10).iterrows()]
t10_dict = {}
for (toban, teiban, v), grp in tqdm(df_seiseki_past.groupby(['登録番号', '艇番', '場']), desc="当地過去10走"):
    t10_dict[(str(toban), int(teiban), str(v))] = [{
        'ym': pd.to_datetime(r['Date'], format='%y%m%d').strftime('%y/%m'),
        'course': int(r.get('コース', teiban)) if pd.notna(r.get('コース')) else teiban,
        'st': str(r['ST_表示']),
        'rank': get_maru_rank(r['着順'], r.get('レース種別', '一般'), r['R'], r.get('DayNum', ''))
    } for _, r in grp.tail(10).iterrows()]

# 選手の節間履歴の区切り判定：場が変わったら区切る／DayNumが本当に巻き戻ったら区切る
temp_s = df_seiseki.copy().sort_values(['登録番号', 'Date', 'R'])
temp_s['Date_dt'] = pd.to_datetime(temp_s['Date'], format='%y%m%d')

def _daynum_sort_val(dn):
    s = str(dn).strip()
    if s == '最終日': return 90
    if s == '優勝戦日': return 91
    try:
        return int(s)
    except ValueError:
        return 50

temp_s['_dn_val']     = temp_s['DayNum'].apply(_daynum_sort_val)
temp_s['_prev_venue'] = temp_s.groupby('登録番号')['場'].shift(1)
temp_s['_prev_dnval'] = temp_s.groupby('登録番号')['_dn_val'].shift(1)
temp_s['_prev_date']  = temp_s.groupby('登録番号')['Date_dt'].shift(1)
temp_s['_gap_days']   = (temp_s['Date_dt'] - temp_s['_prev_date']).dt.days

temp_s['_new_setsu'] = (
    temp_s['_prev_venue'].isna()
    | (temp_s['場'] != temp_s['_prev_venue'])
    | (temp_s['_gap_days'] >= 3)
    | (temp_s['_dn_val'] < temp_s['_prev_dnval'])
)
temp_s['setsu_id'] = temp_s['_new_setsu'].cumsum()
temp_s.drop(columns=['_dn_val', '_prev_venue', '_prev_dnval', '_prev_date', '_gap_days', '_new_setsu'], inplace=True)

history_dict = {}
for toban, group in tqdm(temp_s.groupby('登録番号'), desc="選手節間履歴"):
    setsu_list = []
    for setsu_id, s_group in group.groupby('setsu_id'):
        races = [{
            'date': r['Date_dt'],
            'R': int(r['R']),
            'teiban': int(r['艇番']),
            'course': int(r['コース']) if pd.notna(r['コース']) else int(r['艇番']),
            'st': str(r['ST_表示']),
            'rank': get_maru_rank(r['着順'], r.get('レース種別', '一般'), r['R'], r.get('DayNum', ''))
        } for _, r in s_group.iterrows()]
        setsu_list.append({'venue': s_group['場'].iloc[0], 'grade': s_group.get('グレード', pd.Series(['一般'])).iloc[0], 'date_max': s_group['Date_dt'].max(), 'date_min': s_group['Date_dt'].min(), 'races': races})
    setsu_list.sort(key=lambda x: x['date_max'], reverse=True)
    history_dict[str(toban)] = setsu_list
# ==========================================
# 🌟 6. ガチ・バックテスト（キャッシュ崩壊防止ガード付き）
# ==========================================
backtest_cache_file = os.path.join(CACHE_DIR, f'backtest_v25_{target_date}.json')
cache_loaded = False
if os.path.exists(backtest_cache_file):
    try:
        print(f"\n✅ 本日({target_date})分のバックテスト(v25)を発見しました。読み込み中...")
        with open(backtest_cache_file, 'r') as f:
            b_stats = json.load(f)
        if '1d' in b_stats and 'history' in b_stats['1d']:
            cache_loaded = True
        else:
            print("⚠️ キャッシュの構造が古いため再計算します。")
            cache_loaded = False
    except:
        print("⚠️ キャッシュ破損のため、再計算を実施します。")
        cache_loaded = False
if not os.path.exists(backtest_cache_file) or not cache_loaded:
    print("\n--- 📊 過去の実績と高配当ランキングをシミュレーション中 ---")
    b_stats = {
        '1d': {'cost': 0, 'return': 0, 'hits': 0, 'races': 0, 'history': []},
        '7d': {'cost': 0, 'return': 0, 'hits': 0, 'races': 0, 'history': []},
        '30d': {'cost': 0, 'return': 0, 'hits': 0, 'races': 0, 'history': []}
    }
    target_dt_obj = pd.to_datetime(target_date, format='%y%m%d')
    date_7d_str = (target_dt_obj - pd.Timedelta(days=7)).strftime('%y%m%d')
    date_30d_str = (target_dt_obj - pd.Timedelta(days=30)).strftime('%y%m%d')
    past_df = combined_b_df[(combined_b_df['Date'] >= date_30d_str) & (combined_b_df['Date'] < target_date)]
    latest_past_date = past_df['Date'].max() if not past_df.empty else ""
    for (d_str, v, r), r_group in tqdm(past_df.groupby(['Date', '場', 'R']), desc="バックテスト解析"):
        if '三連単_combo' not in r_group.columns or '三連単_payout' not in r_group.columns: continue
        actual_combo = str(r_group['三連単_combo'].iloc[0]).strip()
        actual_payout_str = str(r_group['三連単_payout'].iloc[0]).replace(',', '').strip()
        if actual_combo in ['nan', '-', ''] or not actual_payout_str.isdigit(): continue
        actual_payout = int(actual_payout_str)
        sum_raw_p1 = r_group['prob_1_raw'].sum()
        if sum_raw_p1 <= 0: sum_raw_p1 = 1.0
        p1_rates = {int(row['艇番']): float(row['prob_1_raw'] / sum_raw_p1) for _, row in r_group.iterrows()}

        sorted_by_p1 = sorted(p1_rates.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_by_p1) < 6: continue

        b1, b2, b3, b4, b5, b6 = [x[0] for x in sorted_by_p1[:6]]
        p1_top = p1_rates[b1]
        is_sg = ('SG' in str(r_group['グレード'].iloc[0]))
        if p1_top >= 0.60:
            if is_sg:
                forms = [{"label": "🎯 SG本線", "pos1": [b1], "pos2": get_unique_pos([b2, b3, b4], [b1]), "pos3": get_unique_pos([b2, b3, b4, b5], [b1])},
                         {"label": "🔥 SG逆転", "pos1": get_unique_pos([b2, b3], [b1]), "pos2": [b1], "pos3": get_unique_pos([b2, b3, b4, b5], [b1])}]
            else:
                forms = [{"label": "🎯 本線", "pos1": [b1], "pos2": get_unique_pos([b2, b3], [b1]), "pos3": get_unique_pos([b2, b3, b4, b5], [b1, b2, b3])},
                         {"label": "🛡️ 押さえ", "pos1": [b1], "pos2": [b4], "pos3": get_unique_pos([b2, b3, b5], [b1, b4])}]
        elif p1_top >= 0.35:
            p1_m1 = [b1, b2]
            p2_m1 = get_unique_pos([b1, b2, b3], [])
            p3_m1 = get_unique_pos([b1, b2, b3, b4], [])
            p1_m2 = [b3, b4]
            p2_m2 = get_unique_pos([b1, b2, b3, b4], [])
            p3_m2 = get_unique_pos([b1, b2, b3, b4], [])
            label1 = "🎯 SG本線" if is_sg else "🎯 本線"
            label2 = "🌪️ SG波乱" if is_sg else "🔥 逆転・穴"
            forms = [{"label": label1, "pos1": p1_m1, "pos2": p2_m1, "pos3": p3_m1},
                     {"label": label2, "pos1": p1_m2, "pos2": p2_m2, "pos3": p3_m2}]
        else:
            p1_m1 = [b1, b2, b3]
            p2_m1 = [b1, b2, b3]
            p3_m1 = get_unique_pos([b1, b2, b3, b4], [])
            p1_m2 = [b4, b5, b6]
            p2_m2 = get_unique_pos([b1, b2, b3], [])
            p3_m2 = get_unique_pos([b1, b2, b3, b4], [])
            label1 = "⚔️ SG混戦" if is_sg else "⚔️ 混戦ボックス"
            label2 = "🔥 夢の万舟" if is_sg else "🔥 大穴"
            forms = [{"label": label1, "pos1": p1_m1, "pos2": p2_m1, "pos3": p3_m1},
                     {"label": label2, "pos1": p1_m2, "pos2": p2_m2, "pos3": p3_m2}]
        bought_combos = set()
        hit = 0
        ret = 0
        hit_label = ""

        for f in forms:
            f_combos = set()
            for p_1 in f['pos1']:
                for p_2 in f['pos2']:
                    if p_1 == p_2: continue
                    for p_3 in f['pos3']:
                        if p_1 == p_3 or p_2 == p_3: continue
                        f_combos.add(f"{p_1}-{p_2}-{p_3}")
            bought_combos.update(f_combos)
            if actual_combo in f_combos and hit == 0:
                hit = 1
                ret = actual_payout
                hit_label = f['label']
        cost = int(len(bought_combos) * 100)
        ret = int(ret)
        hit = int(hit)
        race_num = int(r)
        b_stats['30d']['races'] += 1; b_stats['30d']['cost'] += cost; b_stats['30d']['return'] += ret; b_stats['30d']['hits'] += hit
        if d_str >= date_7d_str:
            b_stats['7d']['races'] += 1; b_stats['7d']['cost'] += cost; b_stats['7d']['return'] += ret; b_stats['7d']['hits'] += hit
        if d_str == latest_past_date:
            b_stats['1d']['races'] += 1; b_stats['1d']['cost'] += cost; b_stats['1d']['return'] += ret; b_stats['1d']['hits'] += hit

        if hit > 0:
            cat = "本線" if "本線" in hit_label else ("穴" if any(x in hit_label for x in ["穴", "逆転", "波乱", "万舟"]) else "押さえ")
            rec = {'date': str(d_str), 'venue': str(v), 'race': race_num, 'combo': str(actual_combo), 'payout': ret, 'label': str(cat)}
            b_stats['30d']['history'].append(rec)
            if d_str >= date_7d_str: b_stats['7d']['history'].append(rec)
            if d_str == latest_past_date: b_stats['1d']['history'].append(rec)
    with open(backtest_cache_file, 'w') as f:
        json.dump(b_stats, f)
    print("✅ 新方式(v25)でのガチ実績集計が完了しました！")
# 🏆 高配当ランキングの抽出
ai_stats_data = {'high_payouts': {'1d': {}, '7d': {}, '30d': {}}}
for k in ['1d', '7d', '30d']:
    c = b_stats[k]['cost']; r = b_stats[k]['return']; h = b_stats[k]['hits']; rc = b_stats[k]['races']
    ai_stats_data[k] = {
        'hit_rate': f"{(h / rc * 100):.1f}" if rc > 0 else "0.0",
        'return_rate': f"{(r / c * 100):.1f}" if c > 0 else "0.0",
    }
    hist = b_stats[k].get('history', [])
    for cat in ["本線", "押さえ", "穴"]:
        cat_hits = [x for x in hist if x['label'] == cat]
        cat_hits.sort(key=lambda x: x['payout'], reverse=True)
        ai_stats_data['high_payouts'][k][cat] = cat_hits[:3]
# ==========================================
# 🌟 7. 最終パッケージング時のグレード補正
# ==========================================
def get_refined_grade(tournament_name, original_grade):
    name_clean = str(tournament_name).upper().replace(' ', '').replace(' ', '')
    sg_list = ["ボートレースクラシック", "ボートレースオールスター", "グランドチャンピオン", "オーシャンカップ", "ボートレースメモリアル", "ボートレースダービー", "チャレンジカップ", "グランプリシリーズ", "グランプリ"]
    is_sg = False
    has_round = bool(re.search(r'第[0-9〇一二三四五六七八九十百]+回', name_clean))
    for sg in sg_list:
        if sg in name_clean and has_round:
            is_sg = True
            break
    if is_sg: return "SG"

    if "レディースチャレンジカップ" in name_clean or "レディースCC" in name_clean: return "G2"
    if re.search(r'G2|GⅡ|GII|モーターボート大賞|MB大賞|モーターボート誕生祭|ボートレース甲子園|甲子園|レディースオールスター|秩父宮妃記念杯', name_clean):
        return "G2"

    g1_list = ["赤城雷神杯", "戸田プリムローズ", "江戸川大賞", "トーキョー・ベイ・カップ", "トーキョーベイカップ", "ウェイキーカップ", "浜名湖賞", "オールジャパン竹島特別", "トコタンキング決定戦", "ツッキー王座決定戦", "北陸艇王決戦", "びわこ大賞", "太閤賞", "尼崎センプルカップ", "大渦大賞", "京極賞", "児島キングカップ", "宮島チャンピオンカップ", "徳山クラウン争奪戦", "競帝王決定戦", "全日本覇者決定戦", "全日本王座決定戦", "福岡チャンピオンカップ", "全日本王者決定戦", "海の王者決定戦", "地区選手権", "ダイヤモンドカップ", "高松宮記念", "ヤングダービー", "マスターズチャンピオン", "レディースチャンピオン", "クイーンズクライマックス", "BBCトーナメント", "スピードクイーンメモリアル", "名人戦"]
    if any(g1 in name_clean for g1 in g1_list): return "G1"
    elif re.search(r'G1|GⅠ', name_clean):
        if not re.search(r'市制|町制|区制|村制|BTS|チケットショップ|ナイター', name_clean): return "G1"

    g3_hit = any(g3 in name_clean for g3 in ["オールレディース", "イースタンヤング", "ウエスタンヤング", "企業杯", "マスターズリーグ", "サッポロビールカップ", "キリンカップ", "アサヒビールカップ", "サントリーカップ"])
    if g3_hit and "マスターズ" in name_clean and "マスターズリーグ" not in name_clean:
        g3_hit = False
    if g3_hit or re.search(r'G3|GⅢ|GIII', name_clean): return "G3"

    return "一般"
# ==========================================
# 🌟 8. Web用 新データ構造 ビルド
# ==========================================
print("\n--- 全日程のデータをパッケージング中 ---")
web_data = {"global_stats": ai_stats_data}
target_date_dt = pd.to_datetime(target_date, format='%y%m%d')
for venue, v_group in tqdm(combined_b_df.groupby('場'), desc="データ構築"):
    df_v_test = df_test[df_test['場'] == venue] if not df_test.empty else pd.DataFrame()
    v_dates = sorted(combined_b_df[combined_b_df['場'] == venue]['Date'].unique())
    setsu_dates = [target_date]
    idx = v_dates.index(target_date) if target_date in v_dates else -1

    if idx != -1:
        target_t_name = combined_b_df[(combined_b_df['場'] == venue) & (combined_b_df['Date'] == target_date)]['レース名'].iloc[0]
        for i in range(idx-1, -1, -1):
            dt_curr = pd.to_datetime(setsu_dates[0], format='%y%m%d')
            dt_prev = pd.to_datetime(v_dates[i], format='%y%m%d')
            prev_t_name = combined_b_df[(combined_b_df['場'] == venue) & (combined_b_df['Date'] == v_dates[i])]['レース名'].iloc[0]

            if pd.notna(target_t_name) and pd.notna(prev_t_name) and target_t_name != "" and target_t_name == prev_t_name:
                setsu_dates.insert(0, v_dates[i])
                continue

            if (dt_curr - dt_prev).days <= 2:
                curr_d_val = combined_b_df[(combined_b_df['場'] == venue) & (combined_b_df['Date'] == setsu_dates[0])]['DayNum'].iloc[0]
                prev_d_val = combined_b_df[(combined_b_df['場'] == venue) & (combined_b_df['Date'] == v_dates[i])]['DayNum'].iloc[0]
                try: curr_d = int(curr_d_val)
                except: curr_d = 99
                try: prev_d = int(prev_d_val)
                except: prev_d = 99

                if prev_d < curr_d: setsu_dates.insert(0, v_dates[i])
                else: break
            else:
                break

    current_day_val = df_v_test['DayNum'].iloc[0] if not df_v_test.empty and 'DayNum' in df_v_test.columns else 1
    try: current_day = int(current_day_val)
    except: current_day = len(setsu_dates)

    web_data[venue] = {"current_day": current_day, "venue_motors": venue_motor_lists.get(venue, []), "days": {}}

    for past_date_str in setsu_dates:
        df_day_b = combined_b_df[(combined_b_df['場'] == venue) & (combined_b_df['Date'] == past_date_str)]
        if df_day_b.empty: continue

        d_val = df_day_b['DayNum'].iloc[0] if 'DayNum' in df_day_b.columns else (setsu_dates.index(past_date_str) + 1)
        try: d = int(d_val)
        except: d = setsu_dates.index(past_date_str) + 1

        web_data[venue]["days"][str(d)] = {}

        for race, r_group in df_day_b.groupby('R'):
            race_str = str(race)

            raw_grade = str(r_group['グレード'].iloc[0]) if 'グレード' in r_group.columns and pd.notna(r_group['グレード'].iloc[0]) else "一般"
            t_name = r_group['レース名'].iloc[0] if 'レース名' in r_group.columns and pd.notna(r_group['レース名'].iloc[0]) else ""
            grade_str = get_refined_grade(t_name, raw_grade)
            is_sg = (grade_str == "SG")

            sum_raw_p1 = r_group['prob_1_raw'].sum()
            if sum_raw_p1 <= 0: sum_raw_p1 = 1.0
            p1_rates_rounded = {int(row['艇番']): round(float(row['prob_1_raw'] / sum_raw_p1) * 100.0, 1) for _, row in r_group.iterrows()}
            total_rounded = sum(p1_rates_rounded.values())
            diff = round(100.0 - total_rounded, 1)
            if diff != 0 and len(p1_rates_rounded) > 0:
                p1_rates_rounded[max(p1_rates_rounded, key=p1_rates_rounded.get)] = round(p1_rates_rounded[max(p1_rates_rounded, key=p1_rates_rounded.get)] + diff, 1)

            boat_probs = {int(row['艇番']): {'p1': p1_rates_rounded[int(row['艇番'])] / 100.0, 'p3_given': {int(i): float(row[f'prob_3_given_{i}']) for i in range(1, 7) if f'prob_3_given_{i}' in row}} for _, row in r_group.iterrows()}

            sorted_by_p1 = sorted(boat_probs.items(), key=lambda item: item[1]['p1'], reverse=True)
            b1 = sorted_by_p1[0][0] if len(sorted_by_p1) > 0 else 1
            b2 = sorted_by_p1[1][0] if len(sorted_by_p1) > 1 else (b1 % 6) + 1
            b3 = sorted_by_p1[2][0] if len(sorted_by_p1) > 2 else (b2 % 6) + 1
            b4 = sorted_by_p1[3][0] if len(sorted_by_p1) > 3 else (b3 % 6) + 1
            b5 = sorted_by_p1[4][0] if len(sorted_by_p1) > 4 else (b4 % 6) + 1
            b6 = sorted_by_p1[5][0] if len(sorted_by_p1) > 5 else (b5 % 6) + 1

            p1_top = boat_probs[b1]['p1'] if b1 in boat_probs else 0

            if p1_top >= 0.60:
                ai_confidence = "鉄板レース"
                if is_sg:
                    p1_sg1 = [b1]
                    p2_sg1 = get_unique_pos([b2, b3, b4], p1_sg1)
                    p3_sg1 = get_unique_pos([b2, b3, b4, b5], p1_sg1)
                    p1_sg2 = get_unique_pos([b2, b3], [b1])
                    p2_sg2 = [b1]
                    p3_sg2 = get_unique_pos([b2, b3, b4, b5], [b1])
                    formations = [
                        {"label": "🎯 SG本線", "pos1": p1_sg1, "pos2": p2_sg1, "pos3": p3_sg1, "pts": calc_pts(p1_sg1, p2_sg1, p3_sg1)},
                        {"label": "🔥 SG逆転", "pos1": p1_sg2, "pos2": p2_sg2, "pos3": p3_sg2, "pts": calc_pts(p1_sg2, p2_sg2, p3_sg2)}
                    ]
                else:
                    p1_m1 = [b1]
                    p2_m1 = get_unique_pos([b2, b3], p1_m1)
                    p3_m1 = get_unique_pos([b2, b3, b4, b5], [b1, b2, b3])
                    p1_m2 = [b1]
                    p2_m2 = get_unique_pos([b4], p1_m2)
                    p3_m2 = get_unique_pos([b2, b3, b5], [b1, b4])
                    formations = [
                        {"label": "🎯 本線", "pos1": p1_m1, "pos2": p2_m1, "pos3": p3_m1, "pts": calc_pts(p1_m1, p2_m1, p3_m1)},
                        {"label": "🛡️ 押さえ", "pos1": p1_m2, "pos2": p2_m2, "pos3": p3_m2, "pts": calc_pts(p1_m2, p2_m2, p3_m2)}
                    ]
            elif p1_top >= 0.35:
                ai_confidence = "中穴・波乱含み"
                p1_m1 = [b1, b2]
                p2_m1 = get_unique_pos([b1, b2, b3], [])
                p3_m1 = get_unique_pos([b1, b2, b3, b4], [])
                p1_m2 = [b3, b4]
                p2_m2 = get_unique_pos([b1, b2, b3, b4], [])
                p3_m2 = get_unique_pos([b1, b2, b3, b4], [])
                label1 = "🎯 SG本線" if is_sg else "🎯 本線"
                label2 = "🌪️ SG波乱" if is_sg else "🔥 逆転・穴"
                formations = [
                    {"label": label1, "pos1": p1_m1, "pos2": p2_m1, "pos3": p3_m1, "pts": calc_pts(p1_m1, p2_m1, p3_m1)},
                    {"label": label2, "pos1": p1_m2, "pos2": p2_m2, "pos3": p3_m2, "pts": calc_pts(p1_m2, p2_m2, p3_m2)}
                ]
            else:
                ai_confidence = "大荒れ警戒"
                p1_m1 = [b1, b2, b3]
                p2_m1 = [b1, b2, b3]
                p3_m1 = get_unique_pos([b1, b2, b3, b4], [])
                p1_m2 = [b4, b5, b6]
                p2_m2 = get_unique_pos([b1, b2, b3], [])
                p3_m2 = get_unique_pos([b1, b2, b3, b4], [])
                label1 = "⚔️ SG混戦" if is_sg else "⚔️ 混戦ボックス"
                label2 = "🔥 夢の万舟" if is_sg else "🔥 大穴"
                formations = [
                    {"label": label1, "pos1": p1_m1, "pos2": p2_m1, "pos3": p3_m1, "pts": calc_pts(p1_m1, p2_m1, p3_m1)},
                    {"label": label2, "pos1": p1_m2, "pos2": p2_m2, "pos3": p3_m2, "pts": calc_pts(p1_m2, p2_m2, p3_m2)}
                ]

            boats_info = []
            for _, row in r_group.sort_values(by='艇番').iterrows():
                b = int(row['艇番'])
                toban_str = str(row.get('登録番号', ''))
                konsetsu_days, zenkoku_3, touchi_3 = {}, [], []
                for setsu in history_dict.get(toban_str, []):
                    if setsu['venue'] == venue and (target_date_dt - setsu['date_max']).days <= 7:
                        for pr in setsu['races']:
                            pdn = int((pr['date'] - setsu['date_min']).days + 1)
                            if pdn not in konsetsu_days: konsetsu_days[pdn] = []
                            konsetsu_days[pdn].append({'R': pr['R'], 'teiban': pr['teiban'], 'course': pr['course'], 'st': pr['st'], 'rank': pr['rank']})
                    else:
                        if len(zenkoku_3) < 3: zenkoku_3.append({"venue": setsu['venue'], "grade": setsu['grade'], "period": f"{setsu['date_min'].month}/{setsu['date_min'].day}-{setsu['date_max'].month}/{setsu['date_max'].day}", "rank": "".join([str(pr['rank']) for pr in setsu['races']])})
                        if setsu['venue'] == venue and len(touchi_3) < 3: touchi_3.append({"grade": setsu['grade'], "period": f"{setsu['date_min'].year}/{setsu['date_min'].month}/{setsu['date_min'].day}-{setsu['date_max'].month}/{setsu['date_max'].day}", "rank": "".join([str(pr['rank']) for pr in setsu['races']])})

                while len(zenkoku_3) < 3: zenkoku_3.append({"venue": "", "grade": "", "period": "", "rank": ""})
                while len(touchi_3) < 3: touchi_3.append({"grade": "", "period": "", "rank": ""})

                z_10 = z10_dict.get((toban_str, b), [])
                t_10 = t10_dict.get((toban_str, b, venue), [])
                p3_sum = 0.0
                for i in range(1, 7):
                    if i in boat_probs and b in boat_probs[i]['p3_given']: p3_sum += boat_probs[i]['p1'] * boat_probs[b]['p3_given'][i]
                raw_name = str(row.get('選手名', '不明'))
                formatted_name = format_racer_name(raw_name)
                boats_info.append({
                    "teiban": b, "name": formatted_name, "kyu": {4: 'A1', 3: 'A2', 2: 'B1', 1: 'B2', 0: 'B2'}.get(row.get('級別_num', 0), 'B2'),
                    "is_female": bool(row.get('is_female', 0) == 1), "age": int(row.get('年齢', 0)) if pd.notna(row.get('年齢')) else "-", "branch": str(row.get('支部', '-')),
                    "dob": str(row.get('生年月日', '-')), "height": str(row.get('身長', '-')), "l": int(row.get('L回数', 0)), "f": int(row.get('F回数', 0)), "reg_no": toban_str,
                    "win_rate": f"{float(row.get('全国勝率', 0.0)):.2f}", "ren2_rate": f"{float(row.get('全国2連率', 0.0)):.1f}", "motor_no": int(row.get('モータNO', 0)),
                    "motor_2ren": f"{float(row.get('モータ2連率_raw', row.get('モータ2連率', 0.0))):.1f}",
                    "motor_rank": motor_stats_dict.get(venue, {}).get(int(row.get('モータNO', 0)), {'rank': '-'})['rank'],
                    "motor_yushu": motor_stats_dict.get(venue, {}).get(int(row.get('モータNO', 0)), {'yushu': 0})['yushu'],
                    "motor_yusho": motor_stats_dict.get(venue, {}).get(int(row.get('モータNO', 0)), {'yusho': 0})['yusho'],
                    "p1_rate": p1_rates_rounded.get(b, 0.0), "p3_rate": round(p3_sum * 100, 1), "avg_st": round(float(row.get('平均ST', 0.17)), 2),
                    "z_10": z_10, "t_10": t_10, "konsetsu": konsetsu_days, "zenkoku_3": zenkoku_3[:3], "touchi_3": touchi_3[:3]
                })

            df_day_res = df_seiseki[(df_seiseki['場'] == venue) & (df_seiseki['Date'] == past_date_str) & (df_seiseki['R'] == race)].copy()
            res_obj = None
            if not df_day_res.empty:
                df_day_res['num_rank'] = df_day_res['着順'].apply(safe_rank_convert)
                df_day_res = df_day_res.sort_values('num_rank')
                res_boats = []
                for _, r in df_day_res.iterrows():
                    p_name, p_is_f, p_kyu = "不明", False, "B2"
                    if (past_date_str, venue, int(race), int(r['艇番'])) in b_motor_lookup.index:
                        p_name = format_racer_name(str(b_motor_lookup.loc[(past_date_str, venue, int(race), int(r['艇番'])), '選手名']))
                        p_is_f = bool(b_motor_lookup.loc[(past_date_str, venue, int(race), int(r['艇番'])), 'is_female'] == 1)
                        p_kyu = str(b_motor_lookup.loc[(past_date_str, venue, int(race), int(r['艇番'])), '級別'])

                    clean_result_rank = get_maru_rank(r['着順'], r.get('レース種別', '一般'), r['R'], r.get('DayNum', ''))
                    res_boats.append({'rank': clean_result_rank, 'teiban': int(r['艇番']), 'reg_no': str(r['登録番号']), 'name': p_name, 'kyu': p_kyu, 'course': int(r['コース']) if pd.notna(r['コース']) else int(r['艇番']), 'st': str(r['ST_表示']), 'race_time': str(r.get('レースタイム', '-')), 'is_female': p_is_f})

                f_row = df_day_res.iloc[0]
                w_dir_raw = str(f_row.get('風向', '無風'))
                w_rel_raw = str(f_row.get('相対風向', '無風'))
                display_wind = f"{w_dir_raw} ({w_rel_raw})" if w_dir_raw not in ['無風', '－', ''] else '無風'
                res_obj = {
                    'boats': res_boats, 'kimarite': str(f_row.get('決まり手', '－')), 'weather': str(f_row.get('天候', '晴れ')),
                    'wind_dir': display_wind,
                    'wind_spd': f"{int(safe_float(f_row.get('風速', 0)))}m", 'wave': f"{int(safe_float(f_row.get('波高', 0)))}cm",
                    'payouts': {
                        'sanrentan_combo': str(f_row.get('三連単_combo', '-')), 'sanrentan_payout': str(f_row.get('三連単_payout', '-')), 'sanrentan_pop': str(f_row.get('三連単_pop', '-')),
                        'sanrenfuku_combo': str(f_row.get('三連複_combo', '-')), 'sanrenfuku_payout': str(f_row.get('三連複_payout', '-')), 'sanrenfuku_pop': str(f_row.get('三連複_pop', '-')),
                        'nirentan_combo': str(f_row.get('二連単_combo', '-')), 'nirentan_payout': str(f_row.get('二連単_payout', '-')), 'nirentan_pop': str(f_row.get('二連単_pop', '-')),
                        'nirenfuku_combo': str(f_row.get('二連複_combo', '-')), 'nirenfuku_payout': str(f_row.get('二連複_payout', '-')), 'nirenfuku_pop': str(f_row.get('二連複_pop', '-')),
                        'tanso_combo': str(f_row.get('単勝_combo', '-')), 'tanso_payout': str(f_row.get('単勝_payout', '-')),
                        'fukusho': safe_list(f_row.get('複勝')), 'kakurenfuku': safe_list(f_row.get('拡連複'))
                    }
                }

            race_round_str = str(r_group['レース種別'].iloc[0]) if 'レース種別' in r_group.columns and pd.notna(r_group['レース種別'].iloc[0]) else ""
            deadline_str = str(r_group['締切時間'].iloc[0]) if '締切時間' in r_group.columns and pd.notna(r_group['締切時間'].iloc[0]) else ""
            web_data[venue]["days"][str(d)][race_str] = {
                "grade": grade_str, "tournament_name": t_name,
                "race_round": race_round_str,
                "deadline": deadline_str,
                "ai_confidence": ai_confidence,
                "formations": formations,
                "boats": boats_info, "result": res_obj
            }
print("✅ 【パート2】特徴量計算・学習・集計が完了しました！")

# ==========================================
# 🌟 セル3由来: HTML生成
# ==========================================
try:
    t_year = "20" + target_date[0:2]
    t_month = int(target_date[2:4])
    t_day = int(target_date[4:6])
    display_date = f"{t_year}年{t_month}月{t_day}日"
except NameError:
    display_date = "本日のレース"

json_str = json.dumps(web_data, ensure_ascii=False)

html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>MONSTER AI（仮）</title>
    <style>
        * { box-sizing: border-box; }
        html, body { max-width: 100vw; overflow-x: hidden; margin: 0; padding: 0; }
        body { font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f4f7f6; color: #333; padding: 5px; }
        h1 { text-align: center; color: #1565c0; font-size: 1.5em; margin: 8px 0 2px 0; font-weight: 900; letter-spacing: 1px; font-style: italic; }
        .date-display { text-align: center; font-size: 0.95em; color: #666; margin-bottom: 15px; font-weight: bold; }

        .venue-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 0 5px 15px 5px; }
        .venue-btn { border-radius: 8px; border: 1px solid #ccc; text-align: center; padding: 2px; cursor: pointer; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 75px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: 0.1s; position: relative; overflow: hidden; }
        .venue-btn:active { transform: scale(0.95); }

        .venue-btn.g-ippan { background: #e3f2fd; border: 2px solid #64b5f6; }
        .venue-btn.g-g3 { background: #e8f5e9; border: 2px solid #81c784; }
        .venue-btn.g-g2 { background: #fff3e0; border: 2px solid #ffb74d; }
        .venue-btn.g-g1 { background: #ffebee; border: 2px solid #e57373; box-shadow: 0 3px 6px rgba(229,115,115,0.2); }
        .venue-btn.g-sg { background: #fffde7; border: 2px solid #fff176; box-shadow: 0 4px 8px rgba(251,192,45,0.25); font-weight: bold; }

        .venue-btn.no-race { background: #eeeeee; border: 1px solid #ddd; cursor: not-allowed; opacity: 0.6; box-shadow: none; }
        .v-name { font-size: 1.3em; font-weight: bold; color: #111; margin-bottom: 1px; letter-spacing: 1px;}
        .g-ippan .v-name { color: #1565c0; }
        .g-g3 .v-name { color: #2e7d32; }
        .g-g2 .v-name { color: #ef6c00; }
        .g-g1 .v-name { color: #c62828; }
        .g-sg .v-name { color: #f57f17; }
        .no-race .v-name { color: #777; margin-bottom: 0; font-size: 1.2em; }

        .v-day { font-size: 0.85em; color: #333; font-weight: bold; margin-bottom: 1px; }
        .v-grade { font-size: 0.85em; font-weight: bold; line-height: 1.3; padding-bottom: 2px; letter-spacing: -0.5px; }
        .g-ippan .v-grade { color: #1976d2; }
        .g-g3 .v-grade { color: #388e3c; }
        .g-g2 .v-grade { color: #f57c00; }
        .g-g1 .v-grade { color: #d32f2f; }
        .g-sg .v-grade { color: #fbc02d; font-size: 0.9em; text-shadow: 0px 1px 0px rgba(0,0,0,0.05); }

        .stats-panel { background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 0 5px 20px 5px; border-top: 3px solid #1565c0; }
        .stats-title { font-weight: bold; color: #1565c0; margin-bottom: 10px; font-size: 1.05em; display:flex; align-items:center; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .stat-card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px 4px; text-align: center; }
        .stat-card.highlight { background: #e3f2fd; border-color: #bbdefb; }
        .stat-period { font-size: 0.75em; font-weight: bold; color: #555; margin-bottom: 5px; }
        .highlight .stat-period { color: #1565c0; }
        .stat-metrics { display: flex; justify-content: space-around; align-items: center; }
        .stat-item { flex: 1; }
        .stat-divider { width: 1px; height: 24px; background: #ddd; }
        .stat-label { font-size: 0.65em; color: #777; margin-bottom: 1px; }
        .stat-value { font-size: 1.1em; font-weight: bold; color: #333; letter-spacing: -0.5px; }
        .stat-value.red { color: #d32f2f; }
        .stat-unit { font-size: 0.6em; font-weight: normal; }
        .stat-notice { font-size: 0.65em; color: #999; margin-top: 8px; text-align: right; font-style: italic; }

        .race-tabs { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; margin-bottom: 12px; }
        .race-tab { padding: 4px 0; font-size: 0.85em; font-weight: bold; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer; text-align: center; color: #333; line-height: 1.2; }
        .race-tab.active { background: #1565c0; color: #fff; border-color: #1565c0; }
        .race-tab.disabled { background: #f5f5f5; color: #bbb; cursor: not-allowed; border-color: #eee; }
        .race-tab .deadline { color: #d32f2f; font-size: 0.82em; font-weight: normal; }
        .race-tab.active .deadline { color: #ffcdd2; }
        .race-tab.disabled .deadline { color: #bbb; }

        .venue-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
        .back-btn { background: #555; color: #fff; border: none; border-radius: 5px; padding: 8px 12px; font-weight: bold; cursor: pointer; font-size: 0.9em; }
        .selected-v-title { font-size: 1.1em; font-weight: bold; color: #333; }
        .card { background: white; padding: 10px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px; width: 100%; }

        #race-grade-name { font-size: 0.85em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; width: 100%; text-align: center; letter-spacing: -0.5px; color: #333; margin-bottom: 8px; }

        .pred-box { padding: 10px; border-radius: 5px; margin-bottom: 8px; border: 1px solid transparent; }
        .pred-title { font-weight: bold; font-size: 0.95em; }
        .pos-group { display: flex; flex-wrap: nowrap; gap: 1px; }

        .sub-tabs { display: flex; width: 100%; border-radius: 6px; overflow: hidden; border: 2px solid #1565c0; margin-top: 15px; margin-bottom: 10px; }
        .sub-tab-btn { flex: 1; text-align: center; padding: 8px 0; font-size: 0.95em; font-weight: bold; cursor: pointer; background: #fff; color: #1565c0; border: none; border-right: 2px solid #1565c0; outline: none; transition: 0.2s; }
        .sub-tab-btn:last-child { border-right: none; }
        .sub-tab-btn.active { background: #1565c0; color: #fff; }

        table { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 0.72em; background: #fff; }
        th, td { border: 1px solid #ddd; padding: 3px 0px; text-align: center; vertical-align: middle; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; letter-spacing: -0.8px; }
        th { background-color: #f8f9fa; color: #555; line-height: 1.15; }
        #view-shusso th { font-size: 0.75em; }

        .player-name { color: #1565c0; font-weight: bold; cursor: pointer; text-decoration: underline; display: block; padding: 0; margin: 0; }

        .female-name {
            color: #ff3366 !important;
            text-shadow:
                1px 1px 0 #fff, -1px -1px 0 #fff,
                1px -1px 0 #fff, -1px 1px 0 #fff,
                0px 1px 0 #fff, 0px -1px 0 #fff,
                1px 0px 0 #fff, -1px 0px 0 #fff !important;
        }

        .two-line { font-size: 0.82em; line-height: 1.15; }

        .bg-1 { background: #ffffff !important; color: #333333 !important; font-weight: bold; border: 1px solid #ccc !important; }
        .bg-2 { background: #333333 !important; color: #ffffff !important; font-weight: bold; }
        .bg-3 { background: #e53935 !important; color: #ffffff !important; font-weight: bold; }
        .bg-4 { background: #1e88e5 !important; color: #ffffff !important; font-weight: bold; }
        .bg-5 { background: #fdd835 !important; color: #333333 !important; font-weight: bold; }
        .bg-6 { background: #43a047 !important; color: #ffffff !important; font-weight: bold; }
        .bg-diff { background-color: #ffffff !important; color: #333 !important; font-weight: bold; }

        .player-tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        .player-tab-btn { padding: 5px 2px; border: none; cursor: pointer; border-radius: 4px; font-size: 0.8em; width: calc(33.33% - 4px); justify-content: center; text-align: center; }
        .player-tab-btn.active { outline: 2px solid #ff9800; opacity: 1; }
        .player-tab-btn:not(.active) { opacity: 0.5; }
        .detail-section-title { background: #eee; padding: 4px 8px; font-weight: bold; margin-top: 12px; margin-bottom: 4px; border-left: 4px solid #555; font-size: 0.85em; }

        .row-highlight { background-color: #fff9c4 !important; }

        .day-tabs-container { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; margin-bottom: 8px; background: #e0e0e0; padding: 3px; border-radius: 6px; }
        .day-tab { padding: 5px 0; font-size: 0.75em; font-weight: bold; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer; text-align: center; color: #333; }
        .day-tab.active { background: #1565c0; color: #fff; border-color: #1565c0; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
        .day-tab:disabled { color: #aaa; cursor: not-allowed; opacity: 0.5; background: #dcdcdc; border-color: #ccc; box-shadow: none; }

        .disclaimer-box { margin-top: 25px; padding: 12px; text-align: left; font-size: 0.72em; color: #666; background: #eee; border-radius: 6px; line-height: 1.4; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>MONSTER AI（仮）</h1>
    <div class="date-display">###DATE###</div>

    <div id="home-view">
        <div id="venue-grid" class="venue-grid"></div>

        <div class="stats-panel" id="stats-panel-container">
            </div>

        <div class="disclaimer-box" style="margin: 5px;">
            <b>免責事項・注意事項</b><br>
            当サイトは、ボートレースの過去成績データを機械学習（AI）によって独自に分析した予測情報を掲載しています。情報の正確性には万全を期しておりますが、主催者発表のデータと必ずご照合ください。当サイトの情報を利用したことにより生じたいかなる損害・損失についても、運営者は一切の責任を負いません。舟券の購入は、必ずご自身の判断と責任において行ってください。（投票券の購入は20歳以上になってから）
        </div>
    </div>

    <div id="venue-view" style="display: none;">
        <div class="venue-header">
            <button class="back-btn" onclick="backToHome()">場選択へ戻る</button>
            <div id="selected-v-title" class="selected-v-title"></div>
        </div>

        <div id="day-tabs" class="day-tabs-container"></div>
        <div id="race-tabs" class="race-tabs"></div>

        <div id="race-content" class="card" style="display: none;">
            <div id="prediction-header-block">
                <div style="margin-bottom: 12px;">
                    <div id="race-grade-name"></div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #1565c0; padding-bottom: 4px; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <h2 id="race-title" style="margin: 0; font-size: 1.4em; color: #1565c0;"></h2>
                            <div id="ai-confidence-badge" style="font-size: 0.8em; font-weight: bold; padding: 2px 6px; border-radius: 4px; margin-left:6px;"></div>
                        </div>
                        <div id="race-deadline" style="font-size: 0.95em; color: #d32f2f; font-weight: bold;"></div>
                    </div>
                </div>

                <div id="formations-container"></div>
            </div>

            <div class="sub-tabs" id="main-sub-tabs">
                <button id="tab-shusso" class="sub-tab-btn active" onclick="switchView('shusso')">出走表</button>
                <button id="tab-seiseki" class="sub-tab-btn" onclick="switchView('seiseki')">競走成績</button>
                <button id="tab-result" class="sub-tab-btn" onclick="switchView('result')">結果</button>
                <button id="tab-motor" class="sub-tab-btn" onclick="switchView('motor')">モーター</button>
            </div>

            <div id="view-shusso">
                <table>
                    <thead>
                        <tr>
                            <th style="width:5%">枠</th><th style="width:23%">選手名</th><th style="width:5%">級</th>
                            <th style="width:9%">年齢<br>支部</th><th style="width:6%">F<br>L</th><th style="width:10%">勝率<br>2連率</th>
                            <th style="width:10%">モーター<br>2連率</th><th style="width:16%">AI予想<br>1着率</th><th style="width:16%">AI予想<br>3連率</th>
                        </tr>
                    </thead>
                    <tbody id="boat-table-body"></tbody>
                </table>
            </div>

            <div id="view-seiseki" style="display: none;">
                <div class="player-tabs" id="player-tabs-container"></div>
                <div id="inline-player-info" style="margin-bottom: 8px; color:#333; line-height:1.4;"></div>
                <div id="inline-tables-container"></div>
            </div>

            <div id="view-motor" style="display: none;"></div>
            <div id="view-result" style="display: none;"></div>

            <div class="disclaimer-box">
                <b>免責事項・注意事項</b><br>
                当サイトは、ボートレースの過去成績データを機械学習（AI）によって独自に分析した予測情報を掲載しています。情報の正確性には万全を期しておりますが、主催者発表のデータと必ずご照合ください。当サイトの情報を利用したことにより生じたいかなる損害・損失についても、運営者は一切の責任を負いません。舟券の購入は、必ずご自身の判断と責任において行ってください。（投票券の購入は20歳以上になってから）
            </div>
        </div>
    </div>

    <script>
        const webData = ###JSON_DATA###;

        const gStats = webData.global_stats || {
            '1d': {hit_rate: '0.0', return_rate: '0.0'},
            '7d': {hit_rate: '0.0', return_rate: '0.0'},
            '30d': {hit_rate: '0.0', return_rate: '0.0'},
            high_payouts: {'1d': {}, '7d': {}, '30d': {}}
        };

        const ALL_VENUES = ['桐生', '戸田', '江戸川', '平和島', '多摩川', '浜名湖', '蒲郡', '常滑', '津', '三国', 'びわこ', '住之江', '尼崎', '鳴門', '丸亀', '児島', '宮島', '徳山', '下関', '若松', '芦屋', '福岡', '唐津', '大村'];
        let currentVenue = ""; let currentRace = ""; let currentSubView = "shusso";
        let selectedDay = 1; let currentMotorNo = null; let currentSortCol = ''; let currentSortDir = 'asc';

        function formatGrade(grade) {
            if (!grade) return "一般";
            const g = grade.toUpperCase();
            if (g.includes("SG") || g.includes("ＳＧ")) return "SG";
            if (g.includes("G1") || g.includes("GI") || g.includes("Ｇ１") || g.includes("ＧＩ")) return "GⅠ";
            if (g.includes("G2") || g.includes("GII") || g.includes("Ｇ２") || g.includes("ＧＩＩ")) return "GⅡ";
            if (g.includes("G3") || g.includes("GIII") || g.includes("GⅢ") || g.includes("Ｇ３")) return "GⅢ";
            return grade;
        }

        function renderBoatsBlock(boatArray) {
            return boatArray.map(b => `<span class="bg-${b}" style="display:inline-block; width:18px; height:18px; line-height:18px; text-align:center; border-radius:3px; font-weight:bold; margin:0;">${b}</span>`).join('');
        }

        function formatCombo(comboStr) {
            if(!comboStr || comboStr === '-') return '-';
            return comboStr.split('-').map(num => {
                if (['1','2','3','4','5','6'].includes(num)) {
                    return `<span class="bg-${num}" style="display:inline-block; width:18px; height:18px; line-height:18px; text-align:center; border-radius:3px; font-weight:bold;">${num}</span>`;
                }
                return num;
            }).join('<span style="margin:0 2px; font-weight:bold; color:#777;">-</span>');
        }

        const venueSummary = {};
        Object.keys(webData).forEach(v => {
            if (v === "global_stats") return;
            let venueGrade = "一般";
            if (webData[v] && webData[v].days) {
                const days = Object.keys(webData[v].days);
                if(days.length > 0) {
                    const firstDayObj = webData[v].days[days[0]];
                    const firstRaceKey = Object.keys(firstDayObj)[0];
                    if(firstRaceKey && firstDayObj[firstRaceKey]) {
                        venueGrade = firstDayObj[firstRaceKey].grade || "一般";
                    }
                }
            }
            venueSummary[v] = { grade: formatGrade(venueGrade), day: webData[v].current_day || 1 };
        });

        // 🌟 ランキングUIのレンダリング
        function renderHighPayout(period) {
            ['1d', '7d', '30d'].forEach(p => {
                const el = document.getElementById('btn-po-'+p);
                if(el) {
                    if(p === period) {
                        el.style.background = '#555'; el.style.color = '#fff';
                    } else {
                        el.style.background = '#fff'; el.style.color = '#555';
                    }
                }
            });

            const data = gStats.high_payouts[period];
            if(!data) return;

            let html = '';
            ['本線', '押さえ', '穴'].forEach(cat => {
                const items = data[cat] || [];
                let catColor = cat === '本線' ? '#1565c0' : (cat === '押さえ' ? '#2e7d32' : '#d32f2f');
                let catBg = cat === '本線' ? '#e3f2fd' : (cat === '押さえ' ? '#e8f5e9' : '#ffebee');

                html += `<div style="background:${catBg}; border-left:4px solid ${catColor}; padding:6px 10px; margin-bottom:8px; border-radius:4px;">`;
                html += `<div style="font-weight:bold; color:${catColor}; font-size:0.9em; margin-bottom:4px;">${cat}系 ベスト3</div>`;

                if(items.length === 0) {
                    html += `<div style="font-size:0.75em; color:#777; padding:4px 0;">的中データなし</div>`;
                } else {
                    html += `<table style="font-size:0.75em; width:100%; text-align:left; background:transparent; border:none; margin:0;">`;
                    items.forEach((item, idx) => {
                        let rankIcon = idx === 0 ? '🥇' : (idx === 1 ? '🥈' : '🥉');
                        let dateHtml = '';
                        if (period !== '1d') {
                            let dStr = String(item.date || '');
                            let dispDate = dStr.length === 6 ? `${dStr.slice(2,4)}/${dStr.slice(4,6)}` : dStr;
                            dateHtml = `<br><span style="font-weight:normal; color:#999; font-size:0.85em;">${dispDate}</span>`;
                        }
                        html += `<tr>
                            <td style="border:none; padding:3px 0; width:10%; text-align:center;">${rankIcon}</td>
                            <td style="border:none; padding:3px 0; width:30%; font-weight:bold; color:#444;">${item.venue} ${item.race}R${dateHtml}</td>
                            <td style="border:none; padding:3px 0; width:30%; text-align:center;">${formatCombo(item.combo)}</td>
                            <td style="border:none; padding:3px 0; width:30%; color:#d32f2f; font-weight:bold; text-align:right;">￥${item.payout.toLocaleString()}</td>
                        </tr>`;
                    });
                    html += `</table>`;
                }
                html += `</div>`;
            });
            document.getElementById('high-payout-table-container').innerHTML = html;
        }

        function initApp() {
            const grid = document.getElementById('venue-grid'); let html = '';
            ALL_VENUES.forEach(v => {
                if (venueSummary[v] && webData[v] && Object.keys(webData[v].days || {}).length > 0) {
                    const s = venueSummary[v];
                    let cls = "g-ippan";
                    if (s.grade === "SG") cls = "g-sg";
                    else if (s.grade === "GⅠ") cls = "g-g1";
                    else if (s.grade === "GⅡ") cls = "g-g2";
                    else if (s.grade === "GⅢ") cls = "g-g3";

                    html += `<button class="venue-btn has-race ${cls}" onclick="selectVenue('${v}')"><div class="v-name">${v}</div><div class="v-day">${s.day}日目</div><div class="v-grade">${s.grade}</div></button>`;
                } else { html += `<button class="venue-btn no-race" disabled><div class="v-name">${v}</div></button>`; }
            });
            grid.innerHTML = html;

            document.getElementById('stats-panel-container').innerHTML = `
                <div class="stats-title">📊 AI予想実績</div>
                <div class="stats-grid" id="stats-grid-container">
                    <div class="stat-card highlight">
                        <div class="stat-period">昨日</div>
                        <div class="stat-metrics">
                            <div class="stat-item"><div class="stat-label">的中率</div><div class="stat-value">${gStats['1d'].hit_rate}<span class="stat-unit">%</span></div></div>
                            <div class="stat-divider"></div>
                            <div class="stat-item"><div class="stat-label">回収率</div><div class="stat-value ${parseFloat(gStats['1d'].return_rate) >= 100 ? 'red' : ''}">${gStats['1d'].return_rate}<span class="stat-unit">%</span></div></div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-period">直近1週間</div>
                        <div class="stat-metrics">
                            <div class="stat-item"><div class="stat-label">的中率</div><div class="stat-value">${gStats['7d'].hit_rate}<span class="stat-unit">%</span></div></div>
                            <div class="stat-divider"></div>
                            <div class="stat-item"><div class="stat-label">回収率</div><div class="stat-value ${parseFloat(gStats['7d'].return_rate) >= 100 ? 'red' : ''}">${gStats['7d'].return_rate}<span class="stat-unit">%</span></div></div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-period">直近1ヶ月</div>
                        <div class="stat-metrics">
                            <div class="stat-item"><div class="stat-label">的中率</div><div class="stat-value">${gStats['30d'].hit_rate}<span class="stat-unit">%</span></div></div>
                            <div class="stat-divider"></div>
                            <div class="stat-item"><div class="stat-label">回収率</div><div class="stat-value ${parseFloat(gStats['30d'].return_rate) >= 100 ? 'red' : ''}">${gStats['30d'].return_rate}<span class="stat-unit">%</span></div></div>
                        </div>
                    </div>
                </div>

                <div class="stats-title" style="margin-top:20px;">🏆 AI 高配当的中ランキング</div>
                <div id="high-payout-tabs" class="sub-tabs" style="border: 2px solid #555; margin-top:0;">
                    <button id="btn-po-1d" class="sub-tab-btn active" style="border-right: 2px solid #555; color:#555;" onclick="renderHighPayout('1d')">昨日</button>
                    <button id="btn-po-7d" class="sub-tab-btn" style="border-right: 2px solid #555; color:#555;" onclick="renderHighPayout('7d')">直近1週</button>
                    <button id="btn-po-30d" class="sub-tab-btn" style="color:#555;" onclick="renderHighPayout('30d')">直近1月</button>
                </div>
                <div id="high-payout-table-container"></div>
                <div class="stat-notice" style="margin-top:8px;">※過去レースを現在のAIでバックテストした実測値です</div>
            `;

            renderHighPayout('1d');
        }

        function selectVenue(v) {
            if (!webData[v] || !webData[v].days) return;
            currentVenue = v;
            document.getElementById('home-view').style.display = 'none';
            document.getElementById('venue-view').style.display = 'block';
            selectedDay = webData[v].current_day || 1;
            document.getElementById('selected-v-title').textContent = `${v} (${selectedDay}日目 ${venueSummary[v].grade})`;
            renderDayTabs(); selectDay(selectedDay);
        }

        function renderDayTabs() {
            const container = document.getElementById('day-tabs'); let html = '';
            for(let d=1; d<=7; d++) {
                const hasData = !!webData[currentVenue].days[String(d)];
                const disabledAttr = !hasData ? 'disabled' : '';
                const activeClass = d == selectedDay ? 'active' : '';
                html += `<button id="day-tab-${d}" class="day-tab ${activeClass}" ${disabledAttr} onclick="selectDay(${d})">${d}日目</button>`;
            }
            container.innerHTML = html;
        }

        function selectDay(d) {
            selectedDay = d; renderDayTabs();
            const tabsContainer = document.getElementById('race-tabs'); let tabsHtml = '';
            const availableRaces = Object.keys(webData[currentVenue].days[String(selectedDay)] || {}).map(Number).sort((a,b)=>a-b);

            for(let r=1; r<=12; r++) {
                if(availableRaces.includes(r)) {
                    const deadline = webData[currentVenue].days[String(selectedDay)][String(r)].deadline || "--:--";
                    tabsHtml += `<button id="tab-${r}" class="race-tab" onclick="selectRace('${r}')">${r}R<br><span class="deadline">${deadline}</span></button>`;
                } else { tabsHtml += `<button class="race-tab disabled" disabled>${r}R<br><span class="deadline">--:--</span></button>`; }
            }
            tabsContainer.innerHTML = tabsHtml;

            if(availableRaces.includes(Number(currentRace))) selectRace(currentRace);
            else if(availableRaces.length > 0) selectRace(String(availableRaces[0]));
        }

        function backToHome() {
            document.getElementById('home-view').style.display = 'block';
            document.getElementById('venue-view').style.display = 'none';
            currentVenue = ""; currentRace = ""; window.scrollTo(0, 0);
        }

        function jumpToResult(day, race) {
            if(!race || race === 0) return;
            selectedDay = day; currentRace = String(race);
            renderDayTabs(); selectDay(day);
            document.querySelectorAll('.race-tab').forEach(btn => btn.classList.remove('active'));
            const rTab = document.getElementById(`tab-${race}`); if(rTab) rTab.classList.add('active');
            switchView('result');
        }

        function switchView(view) {
            currentSubView = view;
            ['shusso', 'seiseki', 'result', 'motor'].forEach(t => {
                const el = document.getElementById(`tab-${t}`); const viewEl = document.getElementById(`view-${t}`);
                if(el) el.classList.toggle('active', view === t);
                if(viewEl) viewEl.style.display = view === t ? 'block' : 'none';
            });
            if(view === 'seiseki') updateInlinePlayer(1);
            if(view === 'motor') renderMotorView(null);
            if(view === 'result') renderResultView();
        }

        function selectRace(r) {
            currentRace = String(r); currentMotorNo = null;
            document.querySelectorAll('.race-tab').forEach(btn => btn.classList.remove('active'));
            const rTab = document.getElementById(`tab-${r}`); if(rTab) rTab.classList.add('active');

            const dayObj = webData[currentVenue].days[String(selectedDay)];
            const data = dayObj ? dayObj[currentRace] : null;
            if(!data) { document.getElementById('race-content').style.display = 'none'; return; }

            document.getElementById('prediction-header-block').style.display = 'block';
            const gradeText = formatGrade(data.grade);
            const raceNameHtml = gradeText !== "一般" ? `<span style="color:#d32f2f; border:1px solid #d32f2f; padding:1px 4px; border-radius:3px; margin-right:4px;">${gradeText}</span> <b>${data.tournament_name || ''}</b>` : `<b>${data.tournament_name || ''}</b>`;
            document.getElementById('race-grade-name').innerHTML = raceNameHtml;
            document.getElementById('race-title').textContent = `${currentRace}R ${data.race_round || ''}`;
            document.getElementById('race-deadline').innerHTML = `締切 <span>${data.deadline || '--:--'}</span>`;

            let confColor = data.ai_confidence.includes("鉄板") ? "background: #e8f5e9; color: #2e7d32; border: 1px solid #81c784;" :
                            (data.ai_confidence.includes("中穴") ? "background: #fff3e0; color: #ef6c00; border: 1px solid #ffb74d;" :
                            "background: #ffebee; color: #c62828; border: 1px solid #e57373;");
            document.getElementById('ai-confidence-badge').innerHTML = `<span style="${confColor} padding: 2px 6px; border-radius: 4px;">${data.ai_confidence}</span>`;

            let fHtml = '';
            if (data.formations && data.formations.length > 0) {
                data.formations.forEach((f, idx) => {
                    const bgClass = idx === 0 ? "background: #e3f2fd; border-color: #bbdefb;" : "background: #fff3e0; border-color: #ffe0b2;";
                    const titleColor = idx === 0 ? "color: #1565c0;" : "color: #e65100;";

                    const p1 = renderBoatsBlock(f.pos1);
                    const p2 = renderBoatsBlock(f.pos2);
                    const p3 = renderBoatsBlock(f.pos3);

                    fHtml += `
                    <div class="pred-box" style="${bgClass}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <div class="pred-title" style="${titleColor}">${f.label}</div>
                            <div style="font-size:0.8em; font-weight:bold; color:#555; background:#fff; padding:1px 6px; border-radius:10px; border:1px solid #ccc;">計 ${f.pts}点</div>
                        </div>
                        <div style="font-size: 1.05em; display:flex; align-items:center; flex-wrap:wrap; gap:4px; padding-left:2px;">
                            <div class="pos-group">${p1}</div>
                            <span style="color:#999; font-weight:bold;">－</span>
                            <div class="pos-group">${p2}</div>
                            <span style="color:#999; font-weight:bold;">－</span>
                            <div class="pos-group">${p3}</div>
                        </div>
                    </div>`;
                });
            }
            document.getElementById('formations-container').innerHTML = fHtml;

            const tbody = document.getElementById('boat-table-body'); tbody.innerHTML = '';
            data.boats.forEach(b => {
                const femaleClass = b.is_female ? 'female-name' : '';

                let sName = String(b.name).replace(/[ \\s]+/g, '');
                let dispName = sName.length >= 4 ? sName.substring(0, 2) + ' ' + sName.substring(2) : sName;

                tbody.innerHTML += `<tr><td class="bg-${b.teiban}">${b.teiban}</td><td><span class="player-name ${femaleClass}" onclick="viewPlayerStats(${b.teiban})">${dispName}</span></td><td>${b.kyu}</td><td class="two-line">${b.age}歳<br>${b.branch}</td><td class="two-line">${b.f}<br>${b.l}</td><td class="two-line">${b.win_rate}<br>${b.ren2_rate}%</td><td class="two-line">No.${b.motor_no}<br>${b.motor_2ren}%</td><td style="font-weight:bold; color: #d32f2f;">${b.p1_rate}%</td><td style="font-weight:bold;">${b.p3_rate}%</td></tr>`;
            });

            document.getElementById('player-tabs-container').innerHTML = data.boats.map(b => `<button class="player-tab-btn bg-${b.teiban}" id="player-btn-${b.teiban}" onclick="updateInlinePlayer(${b.teiban})"><b class="${b.is_female?'female-name':''}">${b.name}</b></button>`).join('');
            switchView(currentSubView); document.getElementById('race-content').style.display = 'block';
        }

        function renderResultView() {
            const container = document.getElementById('view-result');
            const dayObj = webData[currentVenue].days[String(selectedDay)];
            const data = dayObj ? dayObj[currentRace] : null; const raceResult = data ? data.result : null;

            if(!raceResult || !raceResult.boats || raceResult.boats.length === 0) {
                container.innerHTML = `<div style="text-align:center; padding:30px 10px; color:#777; font-weight:bold; font-size:1.0em; word-wrap:break-word;">まだこのレースの結果データがありません</div>`;
                return;
            }

            let html = `<table style="font-size:0.75em; margin-bottom:12px; width:100%; table-layout:fixed;">
                    <thead><tr><th style="width:10%">着</th><th style="width:10%">枠番</th><th style="width:28%">選手名</th><th style="width:10%">級別</th><th style="width:10%">進入</th><th style="width:14%">ST</th><th style="width:18%">タイム</th></tr></thead><tbody>`;

            raceResult.boats.forEach(b => {
                const femaleClass = b.is_female ? 'female-name' : '';
                html += `<tr><td>${b.rank}</td><td class="bg-${b.teiban}">${b.teiban}</td><td style="text-align:center;"><span style="font-weight:bold;" class="${femaleClass}">${b.name}</span></td><td>${b.kyu}</td><td>${b.course}</td><td>${b.st}</td><td>${b.race_time}</td></tr>`;
            });
            html += `</tbody></table>`;
            html += `<div style="font-size:0.95em; font-weight:bold; color:#1565c0; text-align:right; margin-bottom:8px; margin-top:-6px;">決まり手：${raceResult.kimarite || '－'}</div>`;
            html += `<div class="detail-section-title">気象情報</div><table style="font-size:0.75em; margin-bottom:18px;"><tr><th>天候</th><td style="font-weight:bold; color:#111;">${raceResult.weather}</td><th>風向</th><td style="font-weight:bold; color:#111;">${raceResult.wind_dir}</td></tr><tr><th>波高</th><td style="font-weight:bold; color:#111;">${raceResult.wave}</td><th>風速</th><td style="font-weight:bold; color:#111;">${raceResult.wind_spd}</td></tr></table>`;

            const p = raceResult.payouts;
            html += `<div class="detail-section-title">払い戻し金</div>
                <table style="font-size:0.75em; text-align:left; width:100%; table-layout:fixed; border-collapse: collapse;">
                    <thead><tr><th style="width:20%; padding-left:4px;">券種</th><th style="width:38%">組合せ</th><th style="width:27%">払戻金</th><th style="width:15%; text-align:center;">人気</th></tr></thead>
                    <tbody>
                        <tr><td style="padding-left:4px; font-weight:bold;">３連単</td><td>${formatCombo(p.sanrentan_combo)}</td><td style="color:#d32f2f; font-weight:bold;">￥${p.sanrentan_payout}</td><td style="text-align:center;">${p.sanrentan_pop}</td></tr>
                        <tr><td style="padding-left:4px; font-weight:bold;">３連複</td><td>${formatCombo(p.sanrenfuku_combo)}</td><td>￥${p.sanrenfuku_payout}</td><td style="text-align:center;">${p.sanrenfuku_pop}</td></tr>
                        <tr><td style="padding-left:4px; font-weight:bold;">２連単</td><td>${formatCombo(p.nirentan_combo)}</td><td>￥${p.nirentan_payout}</td><td style="text-align:center;">${p.nirentan_pop}</td></tr>
                        <tr><td style="padding-left:4px; font-weight:bold;">２連複</td><td>${formatCombo(p.nirenfuku_combo)}</td><td>￥${p.nirenfuku_payout}</td><td style="text-align:center;">${p.nirenfuku_pop}</td></tr>`;

            const kaku = p.kakurenfuku || [];
            kaku.forEach((k, i) => {
                const labelCell = i === 0
                    ? `<td style="padding-left:4px; font-weight:bold; vertical-align:middle;" rowspan="${kaku.length}">拡連複</td>`
                    : '';
                html += `<tr>${labelCell}<td>${formatCombo(k[0])}</td><td>￥${k[1]}</td><td style="text-align:center;">${k[2]}</td></tr>`;
            });
            const fuku = p.fukusho || [];
            fuku.forEach((f, i) => {
                const labelCell = i === 0
                    ? `<td style="padding-left:4px; font-weight:bold; vertical-align:middle;" rowspan="${fuku.length}">複勝</td>`
                    : '';
                html += `<tr>${labelCell}<td>${formatCombo(f[0])}</td><td>￥${f[1]}</td><td style="text-align:center;">-</td></tr>`;
            });
            html += `<tr><td style="padding-left:4px; font-weight:bold;">単勝</td><td>${formatCombo(p.tanso_combo)}</td><td>￥${p.tanso_payout}</td><td style="text-align:center;">-</td></tr>
                    </tbody></table>`;
            container.innerHTML = html;
        }

        function getSortMark(col) { return currentSortCol === col ? (currentSortDir === 'asc' ? ' ▲' : ' ▼') : ''; }

        function renderMotorView(selectedMotorNo = null, sortCol = null) {
            const dayObj = webData[currentVenue].days[String(selectedDay)]; const data = dayObj ? dayObj[currentRace] : null;
            if(!data || !webData[currentVenue].venue_motors) return;

            if (sortCol) {
                if (currentSortCol === sortCol) currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
                else { currentSortCol = sortCol; currentSortDir = (sortCol === 'motor_no') ? 'asc' : 'desc'; }
            }

            let motorList = [...webData[currentVenue].venue_motors];
            if (currentSortCol) {
                motorList.sort((a, b) => {
                    let valA, valB;
                    if (currentSortCol === 'motor_no') { valA = parseInt(a.motor_no); valB = parseInt(b.motor_no); }
                    else if (currentSortCol === 'motor_2ren') { valA = parseFloat(a.motor_2ren); valB = parseFloat(b.motor_2ren); }
                    else if (currentSortCol === 'motor_yushu') { valA = parseInt(a.motor_yushu); valB = parseInt(b.motor_yushu); }
                    else if (currentSortCol === 'motor_yusho') { valA = parseInt(a.motor_yusho); valB = parseInt(b.motor_yusho); }
                    return valA < valB ? (currentSortDir === 'asc' ? -1 : 1) : (valA > valB ? (currentSortDir === 'asc' ? 1 : -1) : 0);
                });
            }

            if(!selectedMotorNo) { const b1 = data.boats.find(b => b.teiban == 1); selectedMotorNo = b1 ? b1.motor_no : (motorList.length > 0 ? motorList[0].motor_no : null); }

            let playerTabsHtml = `<div class="player-tabs" style="margin-bottom: 12px; border-bottom: 2px solid #ddd; padding-bottom: 8px;">`;
            [...data.boats].sort((a,b)=>a.teiban-b.teiban).forEach(b => {
                playerTabsHtml += `<button class="player-tab-btn bg-${b.teiban} ${b.motor_no==selectedMotorNo?'active':''}" onclick="renderMotorView(${b.motor_no}, null)"><b class="${b.is_female?'female-name':''}">${b.name}</b><br><span style="font-size:0.8em; font-weight:normal;">No.${b.motor_no}</span></button>`;
            });
            playerTabsHtml += `</div>`;

            const selectedMotorObj = motorList.find(m => m.motor_no == selectedMotorNo); let detailHtml = '';
            if (selectedMotorObj) {
                detailHtml += `<div class="detail-section-title">モーター No.${selectedMotorObj.motor_no} 過去全節の使用者</div><table style="font-size:0.72em; width:100%; table-layout:fixed;"><thead><tr><th style="width:25%">使用日</th><th style="width:40%">選手名</th><th style="width:15%">級別</th><th style="width:20%">着</th></tr></thead><tbody>`;
                if(selectedMotorObj.motor_history && selectedMotorObj.motor_history.length > 0) {
                    selectedMotorObj.motor_history.forEach(h => {
                        const femaleClass = h.is_female ? 'female-name' : '';
                        detailHtml += `<tr><td>${h.date_str}</td><td style="font-weight:bold;"><span class="${femaleClass}">${h.racer_name}</span></td><td>${h.kyu}</td><td style="font-weight:bold; letter-spacing:1px;">${h.ranks}</td></tr>`;
                    });
                } else { detailHtml += `<tr><td colspan="4">使用履歴がありません</td></tr>`; }
                detailHtml += `</tbody></table>`;
            }

            let mainTableHtml = `<table style="font-size:0.72em; margin-top:20px; margin-bottom:15px; width:100%; table-layout:fixed;"><thead><tr><th style="width:15%">順位</th><th class="sortable-th" style="width:15%" onclick="renderMotorView(${selectedMotorNo}, 'motor_no')">No.<span style="color:#e65100;">${getSortMark('motor_no')}</span></th><th class="sortable-th" style="width:25%" onclick="renderMotorView(${selectedMotorNo}, 'motor_2ren')">2連対率<span style="color:#e65100;">${getSortMark('motor_2ren')}</span></th><th class="sortable-th" style="width:22.5%" onclick="renderMotorView(${selectedMotorNo}, 'motor_yushu')">優出回数<span style="color:#e65100;">${getSortMark('motor_yushu')}</span></th><th class="sortable-th" style="width:22.5%" onclick="renderMotorView(${selectedMotorNo}, 'motor_yusho')">優勝回数<span style="color:#e65100;">${getSortMark('motor_yusho')}</span></th></tr></thead><tbody>`;
            motorList.forEach((m) => {
                mainTableHtml += `<tr class="${m.motor_no==selectedMotorNo?'row-highlight':''}" onclick="renderMotorView(${m.motor_no}, null)" style="cursor:pointer;"><td>${m.motor_rank!=='-'?m.motor_rank+'位':'-'}</td><td style="font-weight:bold; color:#1565c0; text-decoration:underline;">${m.motor_no}</td><td style="font-weight:bold;">${m.motor_2ren}%</td><td>${m.motor_yushu}</td><td>${m.motor_yusho}</td></tr>`;
            });
            mainTableHtml += `</tbody></table>`;
            document.getElementById('view-motor').innerHTML = playerTabsHtml + detailHtml + mainTableHtml;
        }

        function viewPlayerStats(teiban) { switchView('seiseki'); updateInlinePlayer(teiban); }
        function getHistoryCourseClass(course, teiban) { return (!course || course == '-') ? '' : (course != teiban ? 'bg-diff' : `bg-${course}`); }

        function updateInlinePlayer(teiban) {
            const dayObj = webData[currentVenue].days[String(selectedDay)]; const data = dayObj ? dayObj[currentRace] : null;
            if(!data) return; const boat = data.boats.find(b => b.teiban == teiban); if(!boat) return;

            document.querySelectorAll('.player-tab-btn').forEach(btn => btn.classList.remove('active'));
            const pBtn = document.getElementById(`player-btn-${teiban}`); if(pBtn) pBtn.classList.add('active');

            document.getElementById('inline-player-info').innerHTML = `<div style="font-size:1.2em; font-weight:bold; margin-bottom:4px; ${boat.is_female?'color: #ff3366; text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;':''}">${boat.name}</div><div style="font-size:0.85em; color:#444; letter-spacing: -0.2px;">${boat.reg_no}｜${boat.branch}｜${boat.dob.replace(/([A-Z])0+/g, '$1').replace(/[/]0+/g, '/')}</div><div style="font-size:0.85em; color:#444;">${boat.height}cm｜平均ST:${boat.avg_st.toFixed(2)}</div>`;

            let kDaysHtml = '<tr><th style="width:10%"></th>';
            for(let i=1; i<=7; i++) kDaysHtml += `<th colspan="2" style="font-size:0.8em; letter-spacing:-0.5px;">${i}日目</th>`;
            kDaysHtml += '</tr>';

            let kRHtml = '<tr><th>R</th>', kCourseHtml = '<tr><th>進</th>', kStHtml = '<tr><th>ST</th>', kRankHtml = '<tr><th>着</th>';
            for(let d=1; d<=7; d++) {
                const races = boat.konsetsu[d] || []; const r1 = races.length > 0 ? races[0] : null; const r2 = races.length > 1 ? races[1] : null;
                kRHtml += `<td>${r1?r1.R:''}</td><td>${r2?r2.R:''}</td>`;
                kCourseHtml += `<td class="${r1?'bg-'+r1.teiban:''}">${r1?r1.course:''}</td><td class="${r2?'bg-'+r2.teiban:''}">${r2?r2.course:''}</td>`;
                kStHtml += `<td>${r1?r1.st:''}</td><td>${r2?r2.st:''}</td>`;
                kRankHtml += `<td style="font-weight:bold; cursor:pointer; text-decoration:underline; color:#1565c0;" onclick="jumpToResult(${d}, ${r1?r1.R:0})">${r1?r1.rank:''}</td>`;
                kRankHtml += `<td style="font-weight:bold; cursor:pointer; text-decoration:underline; color:#1565c0;" onclick="jumpToResult(${d}, ${r2?r2.R:0})">${r2?r2.rank:''}</td>`;
            }

            let z10_th = '<tr><th style="width:10%"></th>', z10_v = '<tr><th>場</th>', z10_c = '<tr><th>進</th>', z10_st = '<tr><th>ST</th>', z10_r = '<tr><th>着</th>';
            for(let col = 10; col >= 1; col--){
                z10_th += `<th style="width:9%; font-size:0.75em; letter-spacing:-0.8px; padding:2px 0;">${col}走前</th>`; let dataIdx = boat.z_10.length - col;
                if(dataIdx >= 0 && dataIdx < boat.z_10.length) {
                    const r = boat.z_10[dataIdx]; z10_v += `<td>${r.venue}</td>`; z10_c += `<td class="${getHistoryCourseClass(r.course, boat.teiban)}">${r.course}</td>`; z10_st += `<td>${r.st}</td>`; z10_r += `<td style="font-weight:bold;">${r.rank}</td>`;
                } else { z10_v += '<td>-</td>'; z10_c += '<td>-</td>'; z10_st += '<td>-</td>'; z10_r += '<td>-</td>'; }
            }

            let t10_th = '<tr><th style="width:10%"></th>', t10_ym = '<tr><th>年月</th>', t10_c = '<tr><th>進</th>', t10_st = '<tr><th>ST</th>', t10_r = '<tr><th>着</th>';
            for(let col = 10; col >= 1; col--){
                t10_th += `<th style="width:9%; font-size:0.75em; letter-spacing:-0.8px; padding:2px 0;">${col}走前</th>`; let dataIdx = boat.t_10.length - col;
                if(dataIdx >= 0 && dataIdx < boat.t_10.length) {
                    const r = boat.t_10[dataIdx]; t10_ym += `<td style="font-size:0.8em; padding:4px 0;">${r.ym}</td>`; t10_c += `<td class="${getHistoryCourseClass(r.course, boat.teiban)}">${r.course}</td>`; t10_st += `<td>${r.st}</td>`; t10_r += `<td style="font-weight:bold;">${r.rank}</td>`;
                } else { t10_ym += '<td>-</td>'; t10_c += '<td>-</td>'; t10_st += '<td>-</td>'; t10_r += '<td>-</td>'; }
            }

            let z3_rows = boat.zenkoku_3.map((z, i) => !z.venue && !z.grade && !z.period ? `<tr><td>${i+1}節前</td><td></td><td></td><td></td><td></td></tr>` : `<tr><td>${i+1}節前</td><td>${z.venue}</td><td>${formatGrade(z.grade)}</td><td style="font-size:0.85em; color:#555;">${z.period}</td><td style="font-weight:bold; letter-spacing:1px;">${z.rank}</td></tr>`).join('');
            let t3_rows = boat.touchi_3.map((z, i) => !z.grade && !z.period ? `<tr><td>${i+1}節前</td><td></td><td></td><td></td></tr>` : `<tr><td>${i+1}節前</td><td>${formatGrade(z.grade)}</td><td style="font-size:0.85em; color:#555;">${z.period}</td><td style="font-weight:bold; letter-spacing:1px;">${z.rank}</td></tr>`).join('');

            document.getElementById('inline-tables-container').innerHTML = `<div class="detail-section-title">今節成績</div><table style="font-size:0.65em;">${kDaysHtml}${kRHtml}</tr>${kCourseHtml}</tr>${kStHtml}</tr>${kRankHtml}</tr></table><div class="detail-section-title">全国 枠番別直近10走</div><table style="font-size:0.68em;">${z10_th}</tr>${z10_v}</tr>${z10_c}</tr>${z10_st}</tr>${z10_r}</tr></table><div class="detail-section-title">当地 枠番別直近10走</div><table style="font-size:0.68em;">${t10_th}</tr>${t10_ym}</tr>${t10_c}</tr>${t10_st}</tr>${t10_r}</tr></table><div class="detail-section-title">全国 過去3節</div><table style="font-size:0.7em;"><tr><th rowspan="2" style="width:14%">節</th><th colspan="3">レース</th><th rowspan="2" style="width:25%">着</th></tr><tr><th style="width:15%">場</th><th style="width:17%">グレード</th><th style="width:29%">期間</th></tr>${z3_rows}</table><div class="detail-section-title">当地 過去3節</div><table style="font-size:0.7em;"><tr><th rowspan="2" style="width:14%">節</th><th colspan="2">レース</th><th rowspan="2" style="width:25%">着</th></tr><tr><th style="width:20%">グレード</th><th style="width:41%">期間</th></tr>${t3_rows}</table>`;
        }

        window.onload = initApp;
    </script>
</body>
</html>
"""

html_output = html_template.replace("###DATE###", display_date)
html_output = html_output.replace("###JSON_DATA###", json_str)

# 履歴用にファイル名にも日付を入れておく＋公開用は必ずindex.htmlとして保存
dated_filename = os.path.join(BASE_DIR, f"monster_ai_{target_date}.html")
with open(dated_filename, "w", encoding="utf-8") as f:
    f.write(html_output)

publish_path = os.path.join(PUBLISH_DIR, "index.html")
with open(publish_path, "w", encoding="utf-8") as f:
    f.write(html_output)

elapsed = time.time() - start_time
print(f"\n✅ サイト生成完了: {publish_path}")
print(f"⏱ 総処理時間: {elapsed/60:.1f}分")
