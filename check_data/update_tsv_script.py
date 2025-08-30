#!/usr/bin/env python3
"""
TSVファイル更新スクリプト
test_doc_ids_with_fullinfo_tsv.tsvファイルの4桁コードで市場・商品区分が空の行を
data_j.tsvから対応する情報で更新する
"""

import pandas as pd
import sys
from pathlib import Path

def main():
    # ファイルパス
    base_dir = Path("/Users/yushi/work/project/tech_tf/sigfin/EDINET-Bench/check_data/data")
    test_file = base_dir / "test_doc_ids_with_fullinfo_tsv.tsv"
    data_j_file = base_dir / "data_j.tsv"
    
    print("=== TSVファイル更新スクリプト ===")
    print(f"対象ファイル: {test_file}")
    print(f"参照ファイル: {data_j_file}")
    
    try:
        # ファイル読み込み
        print("\n1. ファイル読み込み中...")
        test_df = pd.read_csv(test_file, sep='\t', encoding='utf-8')
        data_j_df = pd.read_csv(data_j_file, sep='\t', encoding='cp932')
        
        print(f"   test_df shape: {test_df.shape}")
        print(f"   data_j_df shape: {data_j_df.shape}")
        
        # 対象行を特定 (4桁コード + 市場・商品区分が空)
        print("\n2. 対象行を特定中...")
        target_mask = (
            test_df['コード'].astype(str).str.match(r'^[0-9]{4}$') & 
            (test_df['市場・商品区分'].isna() | (test_df['市場・商品区分'] == ''))
        )
        
        target_indices = test_df[target_mask].index.tolist()
        target_codes = test_df.loc[target_indices, 'コード'].tolist()
        
        print(f"   対象行数: {len(target_indices)}")
        print(f"   対象コード例: {target_codes[:5]}")
        
        # data_j_dfから更新データを取得
        print("\n3. マッピングデータを作成中...")
        update_columns = [
            '市場・商品区分', '33業種コード', '33業種区分', 
            '17業種コード', '17業種区分', '規模コード', '規模区分'
        ]
        
        # コードをキーとしてマッピング辞書を作成
        mapping_dict = {}
        for _, row in data_j_df.iterrows():
            code = row['コード']
            mapping_dict[code] = {
                '市場・商品区分': row['市場・商品区分'],
                '33業種コード': row['33業種コード'],
                '33業種区分': row['33業種区分'],
                '17業種コード': row['17業種コード'],
                '17業種区分': row['17業種区分'],
                '規模コード': row['規模コード'],
                '規模区分': row['規模区分']
            }
        
        # 更新を実行
        print("\n4. データ更新中...")
        updated_count = 0
        not_found_codes = []
        
        for idx in target_indices:
            code = test_df.loc[idx, 'コード']
            
            if code in mapping_dict:
                # データを更新
                for col in update_columns:
                    test_df.loc[idx, col] = mapping_dict[code][col]
                updated_count += 1
            else:
                not_found_codes.append(code)
        
        print(f"   更新成功: {updated_count}行")
        print(f"   更新失敗: {len(not_found_codes)}行")
        
        if not_found_codes:
            print(f"   見つからなかったコード: {not_found_codes[:10]}")
        
        # ファイルを保存
        print("\n5. ファイル保存中...")
        test_df.to_csv(test_file, sep='\t', encoding='utf-8', index=False)
        print(f"   保存完了: {test_file}")
        
        # 結果サマリー
        print("\n=== 更新完了 ===")
        print(f"総対象行数: {len(target_indices)}")
        print(f"更新成功行数: {updated_count}")
        print(f"更新失敗行数: {len(not_found_codes)}")
        print(f"成功率: {updated_count/len(target_indices)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)