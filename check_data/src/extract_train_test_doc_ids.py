from edinet_bench.logistic import prepare_dataset
import pandas as pd
from pathlib import Path
import json


def main():
    output_path = Path(Path.cwd(), "check_data", "data")

    (
        _,
        train_doc_ids,
        train_prev_year_paths,
        train_current_year_paths,
        train_metadata,
    ) = prepare_dataset(
        task="earnings_forecast",
        split="train",
        use_differential_features=False,
        use_percentage_change_features=False,
        return_metadata=True,
    )
    _, test_doc_ids, test_prev_year_paths, test_current_year_paths, test_metadata = (
        prepare_dataset(
            task="earnings_forecast",
            split="test",
            use_differential_features=False,
            use_percentage_change_features=False,
            return_metadata=True,
        )
    )

    train_company_names = [
        json.loads(meta_data)["会社名"] for meta_data in train_metadata
    ]
    test_company_names = [
        json.loads(meta_data)["会社名"] for meta_data in test_metadata
    ]
    train_start_date = [
        json.loads(meta_data)["当事業年度開始日"] for meta_data in train_metadata
    ]
    test_start_date = [
        json.loads(meta_data)["当事業年度開始日"] for meta_data in test_metadata
    ]
    train_end_date = [
        json.loads(meta_data)["当事業年度終了日"] for meta_data in train_metadata
    ]
    test_end_date = [
        json.loads(meta_data)["当事業年度終了日"] for meta_data in test_metadata
    ]

    df_train_doc_id = pd.DataFrame(
        {
            "train_company_name": train_company_names,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "prev_year_doc_id": [
                val.split("/")[-1].replace(".tsv", "") for val in train_prev_year_paths
            ],
            "current_year_doc_id": [
                val.split("/")[-1].replace(".tsv", "")
                for val in train_current_year_paths
            ],
        }
    )

    df_test_doc_id = pd.DataFrame(
        {
            "test_company_name": test_company_names,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "prev_year_doc_id": [
                val.split("/")[-1].replace(".tsv", "") for val in test_prev_year_paths
            ],
            "current_year_doc_id": [
                val.split("/")[-1].replace(".tsv", "")
                for val in test_current_year_paths
            ],
        }
    )

    df_train_doc_id.to_csv(output_path / "train_doc_ids.tsv", index=False, sep="\t")
    df_test_doc_id.to_csv(output_path / "test_doc_ids.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
