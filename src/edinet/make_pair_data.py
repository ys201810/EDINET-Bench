# 予測対象年と、その前年の有価証券報告書のペアを作成する
import argparse
from pathlib import Path
import pandas as pd
import random
import shutil
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


def index_dir(root: Path, code_re: re.Pattern) -> dict[str, list[tuple[float, Path]]]:
    """root配下の *asr* ファイルを1回走査して、コード -> [(mtime, Path), ...] にまとめる"""
    mapping: dict[str, list[tuple[float, Path]]] = defaultdict(list)
    for p in root.rglob("*asr*"):
        m = code_re.search(p.name)
        if not m:
            continue
        code = m.group(0)
        try:
            mt = p.stat().st_mtime
        except FileNotFoundError:
            # まれに並行更新などで消える場合のガード
            continue
        mapping[code].append((mt, p))
    return mapping


def copy_if_needed(src: Path, dst: Path):
    """既存ファイルがあり、サイズが一致する場合はコピーをスキップ"""
    if dst.exists():
        try:
            if src.stat().st_size == dst.stat().st_size:
                return  # だいたい同一とみなしてコピー省略
        except FileNotFoundError:
            pass
    shutil.copy2(src, dst)


def process_code(
    code: str,
    target_year_map: dict[str, list[tuple[float, Path]]],
    previous_year_map: dict[str, list[tuple[float, Path]]],
    out_path: Path,
) -> tuple:
    """EDINETコードごとのファイルペアをコピー"""
    # 該当ファイルがなければスキップ判定
    target_year_list = target_year_map.get(code, [])
    if not target_year_list:
        return ("no_target_year", code)

    previous_year_list = previous_year_map.get(code, [])
    if not previous_year_list:
        return ("no_previous_year", code)

    if len(previous_year_list) > 1:
        # 「前年が2つ以上」のコードを記録（要調査のシグナル）
        # ※要件によっては target_year_list でもチェック可能
        pass_multi = True
    else:
        pass_multi = False

    # それぞれmtime最大（最新）を採用
    target_year_file = max(target_year_list, key=lambda t: t[0])[1]
    previous_year_file = max(previous_year_list, key=lambda t: t[0])[1]

    # 出力ディレクトリ
    out_files_path = out_path / code
    out_files_path.mkdir(parents=True, exist_ok=True)

    copy_if_needed(target_year_file, out_files_path / target_year_file.name)
    copy_if_needed(previous_year_file, out_files_path / previous_year_file.name)

    return (
        "ok_multi" if pass_multi else "ok",
        code,
        target_year_file.name,
        previous_year_file.name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="予測対象年と、その前年の有価証券報告書のペアを作成"
    )
    parser.add_argument(
        "--target_file",
        type=str,
        required=True,
        help="予測対象年の対象リストTSVファイルのパス",
    )
    parser.add_argument(
        "--target_csv_path",
        type=str,
        required=True,
        help="予測対象年の対象リストCSVファイルのパス",
    )
    parser.add_argument(
        "--prev_csv_path",
        type=str,
        required=True,
        help="前年の対象リストCSVファイルのパス",
    )
    args = parser.parse_args()
    base_path = Path.cwd()
    out_path = Path(
        base_path,
        "data",
        "own",
        args.prev_csv_path[-13:][:4] + "_" + args.target_csv_path[-13:][:4] + "_pair",
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # 予測対象年の対象リスト取得
    pred_target_file = base_path / args.target_file
    df_pred_target_year = pd.read_csv(pred_target_file, sep="\t")

    pred_target_edinet_codes = df_pred_target_year["edinet_code"].unique().tolist()

    # 予測対象年のcsvディレクトリから対象ファイルを抽出
    csv_target_year_path = base_path / args.target_csv_path
    # 前年パス
    csv_previous_year_path = base_path / args.prev_csv_path

    # ---- 1) EDINETコードの巨大OR正規表現を作成（高速化のキモ） ----
    # 長いコードを先頭にマッチさせやすいように長さ降順で並べる
    _codes_sorted = sorted(
        set(map(str, pred_target_edinet_codes)), key=len, reverse=True
    )
    code_re = re.compile("|".join(map(re.escape, _codes_sorted)))

    print("Indexing target year dir ...")
    target_year_map = index_dir(csv_target_year_path, code_re)
    print("Indexing previous year dir ...")
    previous_year_map = index_dir(csv_previous_year_path, code_re)

    over_2_submit_codes: list[str] = []
    no_submit_codes: list[str] = []

    # ---- 2) コピー処理（並列） ----
    # スレッド数は環境に合わせて調整（ディスクが遅い場合は小さめが良い）
    max_workers = 8

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_code, code, target_year_map, previous_year_map, out_path
            ): code
            for code in pred_target_edinet_codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            res = fut.result()
            if res[0] == "no_target_year":
                print(f"skip target year nothing {res[1]}")
                no_submit_codes.append(res[1])
            elif res[0] == "no_previous_year":
                print(f"skip previous year nothing {res[1]}")
                no_submit_codes.append(res[1])
            elif res[0] in ("ok", "ok_multi"):
                _, code, target_name, prev_name = res
                if res[0] == "ok_multi":
                    over_2_submit_codes.append(code)
                print(f"{code}: {target_name}, {prev_name}")

    # 結果のリストは元の変数名を踏襲
    print("over_2_submit_codes:", over_2_submit_codes)
    print("no_submit_codes:", no_submit_codes)


if __name__ == "__main__":
    main()
