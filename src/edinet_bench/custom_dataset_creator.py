"""
Custom Dataset Creator for EDINET-Bench

This module provides functionality to create EDINET-Bench format datasets
from arbitrary EDINET data files.
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datasets import Dataset
from edinet2dataset.parser import parse_tsv, FinancialData
from loguru import logger
import argparse


class CustomDatasetCreator:
    """Creates custom EDINET-Bench format datasets from EDINET data."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset creator.

        Args:
            data_dir: Directory containing EDINET data files (CSV/TSV format)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def find_edinet_files(self, pattern: str = "**/*.csv") -> List[Path]:
        """
        Find EDINET data files in the data directory.

        Args:
            pattern: Glob pattern to match files

        Returns:
            List of file paths
        """
        files = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(files)} EDINET files")
        return files

    def parse_edinet_file(self, file_path: Path) -> Optional[FinancialData]:
        """
        Parse a single EDINET file.

        Args:
            file_path: Path to the EDINET file

        Returns:
            FinancialData object or None if parsing failed
        """
        try:
            # Handle different file formats and encodings
            temp_file = None
            file_to_parse = str(file_path)

            if file_path.suffix.lower() == ".csv":
                # Try to convert CSV to TSV format for edinet2dataset
                temp_file = self._convert_csv_to_tsv(file_path)
                if temp_file:
                    file_to_parse = temp_file
                else:
                    # If conversion fails, try to parse directly
                    logger.warning(
                        f"Could not convert CSV to TSV for {file_path}, trying direct parsing"
                    )

            # Try different encodings
            encodings = ["utf-16", "cp932", "utf-8", "shift-jis"]
            financial_data = None

            for encoding in encodings:
                try:
                    # edinet2dataset's parse_tsv expects UTF-16 encoding
                    financial_data = parse_tsv(file_to_parse)
                    if financial_data:
                        logger.info(
                            f"Successfully parsed {file_path} with encoding {encoding}"
                        )
                        break
                except Exception as e:
                    logger.debug(
                        f"Failed to parse {file_path} with encoding {encoding}: {e}"
                    )
                    continue

            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            if financial_data:
                return financial_data
            else:
                logger.warning(
                    f"Failed to parse {file_path} with all attempted encodings"
                )
                return None

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _convert_csv_to_tsv(self, csv_path: Path) -> Optional[str]:
        """
        Convert CSV file to TSV format for edinet2dataset compatibility.

        Args:
            csv_path: Path to CSV file

        Returns:
            Path to temporary TSV file or None if conversion failed
        """
        import tempfile
        import csv

        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".tsv")

            # Try different encodings for CSV reading
            encodings = ["utf-8", "shift-jis", "cp932", "utf-16"]

            for encoding in encodings:
                try:
                    with open(csv_path, "r", encoding=encoding) as csv_file:
                        # Auto-detect delimiter
                        sample = csv_file.read(1024)
                        csv_file.seek(0)
                        sniffer = csv.Sniffer()
                        delimiter = sniffer.sniff(sample).delimiter

                        reader = csv.reader(csv_file, delimiter=delimiter)

                        with os.fdopen(temp_fd, "w", encoding="utf-16") as tsv_file:
                            writer = csv.writer(tsv_file, delimiter="\t")
                            for row in reader:
                                writer.writerow(row)

                        logger.info(f"Converted CSV to TSV: {csv_path} -> {temp_path}")
                        return temp_path

                except Exception as e:
                    logger.debug(f"Failed to convert CSV with encoding {encoding}: {e}")
                    continue

            # If we get here, conversion failed
            os.close(temp_fd)
            os.remove(temp_path)
            return None

        except Exception as e:
            logger.error(f"Error converting CSV to TSV: {e}")
            return None

    def financial_data_to_dict(
        self, financial_data: FinancialData, doc_id: str, edinet_code: str = None
    ) -> Dict:
        """
        Convert FinancialData to EDINET-Bench format dictionary.

        Args:
            financial_data: Parsed financial data
            doc_id: Document ID
            edinet_code: EDINET code (extracted from meta if not provided)

        Returns:
            Dictionary in EDINET-Bench format
        """
        # Extract EDINET code from meta if not provided
        if not edinet_code and hasattr(financial_data, "meta"):
            edinet_code = financial_data.meta.get("EDINETコード", "UNKNOWN")

        return {
            "meta": json.dumps(financial_data.meta, ensure_ascii=False),
            "summary": json.dumps(financial_data.summary, ensure_ascii=False),
            "bs": json.dumps(financial_data.bs, ensure_ascii=False),
            "pl": json.dumps(financial_data.pl, ensure_ascii=False),
            "cf": json.dumps(financial_data.cf, ensure_ascii=False),
            "text": json.dumps(financial_data.text, ensure_ascii=False),
            "label": 0,  # Default label - should be set based on task requirements
            "naive_prediction": 0,  # Default naive prediction
            "edinet_code": edinet_code or "UNKNOWN",
            "doc_id": doc_id,
            "previous_year_file_path": "",  # To be filled if available
            "current_year_file_path": str(self.data_dir / f"{doc_id}.tsv"),
        }

    def financial_data_to_dict_from_pair(
        self,
        prev_data: FinancialData,
        curr_data: FinancialData,
        prev_file: Path,
        curr_file: Path,
    ) -> Dict:
        """
        Convert pair of FinancialData to EDINET-Bench format dictionary.

        Args:
            prev_data: Previous year financial data
            curr_data: Current year financial data
            prev_file: Previous year file path
            curr_file: Current year file path

        Returns:
            Dictionary in EDINET-Bench format
        """
        # Extract EDINET code from current year meta
        edinet_code = curr_data.meta.get("EDINETコード", "UNKNOWN")
        doc_id = curr_file.stem  # Use current year file name as doc_id

        return {
            "meta": json.dumps(prev_data.meta, ensure_ascii=False),
            "summary": json.dumps(prev_data.summary, ensure_ascii=False),
            "bs": json.dumps(prev_data.bs, ensure_ascii=False),
            "pl": json.dumps(prev_data.pl, ensure_ascii=False),
            "cf": json.dumps(prev_data.cf, ensure_ascii=False),
            "text": json.dumps(prev_data.text, ensure_ascii=False),
            "curr_summary": json.dumps(
                curr_data.summary, ensure_ascii=False
            ),  # Current year summary for comparison
            "label": 0,  # Will be set in create_labels_from_pairs
            "naive_prediction": 0,
            "edinet_code": edinet_code,
            "doc_id": doc_id,
            "previous_year_file_path": str(prev_file),
            "current_year_file_path": str(curr_file),
        }

    def create_file_pairs(self, files: List[Path]) -> List[Tuple[Path, Path]]:
        """
        Create pairs of (previous_year_file, current_year_file) for earnings forecasting.

        Args:
            files: List of EDINET files

        Returns:
            List of (previous_year_file, current_year_file) pairs
        """
        # Group files by EDINET code
        edinet_files = {}

        for file_path in files:
            # Extract EDINET code and date from filename
            # Format: jpcrp030000-asr-001_E05623-000_2019-09-30_01_2019-12-23.csv
            filename = file_path.name
            try:
                parts = filename.split("_")
                if len(parts) >= 3:
                    edinet_code = parts[1]  # E05623-000
                    date_str = parts[2]  # 2019-09-30

                    if edinet_code not in edinet_files:
                        edinet_files[edinet_code] = []
                    edinet_files[edinet_code].append((date_str, file_path))
            except Exception as e:
                logger.warning(f"Could not parse filename {filename}: {e}")
                continue

        # Create pairs for each EDINET code
        pairs = []
        for edinet_code, file_list in edinet_files.items():
            # Sort by date
            file_list.sort(key=lambda x: x[0])

            # Create consecutive pairs
            for i in range(len(file_list) - 1):
                prev_date, prev_file = file_list[i]
                curr_date, curr_file = file_list[i + 1]
                pairs.append((prev_file, curr_file))
                logger.info(
                    f"Created pair for {edinet_code}: {prev_date} -> {curr_date}"
                )

        return pairs

    def create_labels_from_pairs(
        self, pair_data_list: List[Dict], task: str
    ) -> List[Dict]:
        """
        Create labels for the dataset based on file pairs.

        Args:
            pair_data_list: List of data dictionaries created from file pairs
            task: Task type ('earnings_forecast' or 'fraud_detection')

        Returns:
            Updated data list with labels
        """
        prev_none_cnt = 0
        curr_none_cnt = 0
        cant_get_value_cnt = 0
        ng_dict = {}
        if task == "earnings_forecast":
            # For earnings forecast: 1 if current year profit > previous year profit
            for data in pair_data_list:
                try:
                    edinet_code = json.loads(data["meta"])["EDINETコード"]
                    prev_summary = json.loads(data["summary"])
                    curr_summary = json.loads(data["curr_summary"])

                    # Get profit from previous year file (CurrentYear of previous file)
                    prev_profit = prev_summary.get(
                        "親会社株主に帰属する当期純利益", {}
                    ).get("CurrentYear")
                    # Get profit from current year file (CurrentYear of current file)
                    curr_profit = curr_summary.get(
                        "親会社株主に帰属する当期純利益", {}
                    ).get("CurrentYear")

                    if prev_profit and curr_profit:
                        if prev_profit == "－":
                            prev_none_cnt += 1
                            ng_dict[edinet_code] = "prev profit is －"

                        if curr_profit == "－":
                            curr_none_cnt += 1
                            ng_dict[edinet_code] = "curr profit is －"
                        # Convert to float and compare
                        prev_val = float(
                            str(prev_profit).replace(",", "").replace("－", "0")
                        )
                        curr_val = float(
                            str(curr_profit).replace(",", "").replace("－", "0")
                        )
                        data["label"] = 1 if curr_val > prev_val else 0
                        logger.info(
                            f"Label created: {data['doc_id']} - Prev: {prev_val}, Curr: {curr_val}, Label: {data['label']}"
                        )
                    else:
                        cant_get_value_cnt += 1
                        ng_dict[edinet_code] = "profit data not available"
                        data["label"] = 0  # Default if data not available
                        logger.warning(
                            f"Profit data not available for {data['doc_id']}"
                        )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Could not create label for {data['doc_id']}: {e}")
                    ng_dict[edinet_code] = "json_parse_error"
                    data["label"] = 0
            print("prev_none_cnt:", prev_none_cnt)
            print("curr_none_cnt:", curr_none_cnt)
            print("cant_get_value_cnt:", cant_get_value_cnt)

        elif task == "fraud_detection":
            # For fraud detection: This would require additional logic
            # For now, set all to 0 (no fraud detected)
            for data in pair_data_list:
                data["label"] = 0

        return pair_data_list

    def create_dataset(
        self,
        task: str = "earnings_forecast",
        file_pattern: str = "**/*.csv",
        max_files: Optional[int] = None,
    ) -> Dataset:
        """
        Create a custom EDINET-Bench format dataset.

        Args:
            task: Task type ('earnings_forecast' or 'fraud_detection')
            file_pattern: Pattern to match EDINET files
            max_files: Maximum number of files to process (None for all)

        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Creating dataset for task: {task}")

        # Find EDINET files
        files = self.find_edinet_files(file_pattern)
        if max_files:
            files = files[:max_files]

        data_list = []

        if task == "earnings_forecast":
            # For earnings forecasting, we need file pairs
            file_pairs = self.create_file_pairs(files)  # 全データのprev/currペア

            if not file_pairs:
                logger.warning(
                    "No file pairs found for earnings forecasting. Need consecutive years for same company."
                )
                # Fallback to single file processing
                logger.info("Falling back to single file processing")
                return self._create_dataset_single_files(files, task)

            logger.info(f"Found {len(file_pairs)} file pairs for earnings forecasting")

            from tqdm import tqdm

            # file_pairs = file_pairs[:10]

            for prev_file, curr_file in tqdm(file_pairs):
                # Parse both files
                prev_data = self.parse_edinet_file(prev_file)
                curr_data = self.parse_edinet_file(curr_file)

                if prev_data and curr_data:
                    # Convert to EDINET-Bench format using pair
                    data_dict = self.financial_data_to_dict_from_pair(
                        prev_data, curr_data, prev_file, curr_file
                    )
                    data_list.append(data_dict)

            if not data_list:
                raise ValueError("No valid EDINET file pairs found or parsed")

            # Create labels based on pairs
            data_list = self.create_labels_from_pairs(data_list, task)

        else:
            # For other tasks, use single file processing
            data_list = self._create_dataset_single_files(files, task)

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(data_list)

        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset

    def _create_dataset_single_files(self, files: List[Path], task: str) -> List[Dict]:
        """
        Create dataset from single files (fallback or for non-earnings tasks).

        Args:
            files: List of file paths
            task: Task type

        Returns:
            List of data dictionaries
        """
        data_list = []

        for file_path in files:
            # Extract document ID from filename
            doc_id = file_path.stem

            # Parse the file
            financial_data = self.parse_edinet_file(file_path)
            if financial_data:
                # Convert to EDINET-Bench format
                data_dict = self.financial_data_to_dict(financial_data, doc_id)
                data_list.append(data_dict)

        if not data_list:
            raise ValueError("No valid EDINET files found or parsed")

        # Create labels based on task (legacy method for single files)
        # This won't work well for earnings_forecast but kept for compatibility
        if task == "fraud_detection":
            for data in data_list:
                data["label"] = 0  # Default for fraud detection
        else:
            # For earnings_forecast fallback, try to use Prior1Year vs CurrentYear
            for data in data_list:
                try:
                    summary = json.loads(data["summary"])
                    current_profit = summary.get(
                        "親会社株主に帰属する当期純利益", {}
                    ).get("CurrentYear")
                    prior_profit = summary.get(
                        "親会社株主に帰属する当期純利益", {}
                    ).get("Prior1Year")

                    if current_profit and prior_profit:
                        curr_val = float(
                            str(current_profit).replace(",", "").replace("－", "0")
                        )
                        prev_val = float(
                            str(prior_profit).replace(",", "").replace("－", "0")
                        )
                        data["label"] = 1 if curr_val > prev_val else 0
                    else:
                        data["label"] = 0

                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Could not create label for {data['doc_id']}: {e}")
                    data["label"] = 0

        return data_list


def main():
    """Command line interface for creating custom datasets."""
    parser = argparse.ArgumentParser(description="Create custom EDINET-Bench dataset")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing EDINET data files"
    )
    parser.add_argument(
        "--task",
        choices=["earnings_forecast", "fraud_detection"],
        default="earnings_forecast",
        help="Task type",
    )
    parser.add_argument("--output_path", help="Path to save the dataset")
    parser.add_argument(
        "--max_files", type=int, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--file_pattern", default="**/*.csv", help="Pattern to match EDINET files"
    )

    args = parser.parse_args()

    # Create dataset creator
    creator = CustomDatasetCreator(args.data_dir)

    # Create dataset
    dataset = creator.create_dataset(
        task=args.task, file_pattern=args.file_pattern, max_files=args.max_files
    )

    # Save dataset if output path provided
    if args.output_path:
        dataset.save_to_disk(args.output_path)
        logger.info(f"Dataset saved to {args.output_path}")

    # Print sample
    print(f"Dataset created with {len(dataset)} examples")
    if len(dataset) > 0:
        print("Sample data:")
        print(f"Keys: {list(dataset[0].keys())}")
        print(f"EDINET Code: {dataset[0]['edinet_code']}")
        print(f"Doc ID: {dataset[0]['doc_id']}")
        print(f"Label: {dataset[0]['label']}")


if __name__ == "__main__":
    main()
