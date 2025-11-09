import os
import json
from tqdm import tqdm
import argparse
import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
import weave
from loguru import logger
import yaml
from edinet_bench.model import MODEL_TABLE, Model
from dataclasses import dataclass, asdict
from edinet_bench.utils import extract_json_between_markers
from edinet_bench.custom_dataset_creator import CustomDatasetCreator
import pickle


@dataclass
class Result:
    edinet_code: str
    doc_id: str
    label: int
    prob: float | None
    prediction: int | None
    reasoning: str | None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict_data):
        return cls(
            edinet_code=dict_data["edinet_code"],
            doc_id=dict_data["doc_id"],
            label=dict_data["label"],
            prob=dict_data.get("prob", None),
            prediction=dict_data.get("prediction", None),
            reasoning=dict_data.get("reasoning", None),
        )


@weave.op()
def predict(
    prompt: str,
    model: Model,
) -> tuple[float | None, int | None, str | None]:
    """Predict whether current year profit is higher than previous year"""
    response = model.get_completion(prompt)
    logger.info(f"Response: {response}")
    json_data = extract_json_between_markers(response)
    print(json_data)
    if json_data is None:
        return None, None, None
    prob = json_data.get("prob", None)
    prediction = json_data.get("prediction", None)
    reasoning = json_data.get("reasoning", None)
    return prob, prediction, reasoning


@weave.op()
def process_example(example, model: Model, prompt: str, sheets: list[str]) -> Result:
    prompt = prompt + "\n".join(
        [f"{sheet}: {example[sheet]}" for sheet in sheets if sheet in example]
    )

    prob, prediction, reasoning = predict(prompt, model)

    return Result(
        edinet_code=example["edinet_code"],
        doc_id=example["doc_id"],
        label=example["label"],
        prob=prob,
        prediction=prediction,
        reasoning=reasoning,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--num_example", type=int, default=None, help="Number of examples to process"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["fraud_detection", "earnings_forecast"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for sampling"
    )
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--wandb", type=bool, default=True, help="Log to wandb")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model name",
        choices=MODEL_TABLE.keys(),
    )
    parser.add_argument(
        "--sheets",
        type=str,
        nargs="+",
        default=[
            "summary",
            "bs",
            "pl",
            "cf",
        ],
        help="Sheets to include in the prompt. Default: summary, bs, pl, cf",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a financial analyst.",
        help="System prompt for the model.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        help="Path to custom EDINET data directory (if provided, will create custom dataset instead of using HuggingFace)",
    )
    parser.add_argument(
        "--custom_data_pattern",
        type=str,
        default="**/*.csv",
        help="File pattern for custom EDINET data files",
    )
    parser.add_argument(
        "--max_custom_files",
        type=int,
        help="Maximum number of custom files to process",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wandb:
        try:
            weave.init(args.task)
        except Exception as e:
            logger.warning(f"Failed to initialize Weave: {e}")
            logger.info("Continuing without Weave tracking")

    # Load dataset - either from HuggingFace or create custom dataset
    if args.custom_data_path:
        logger.info(f"Creating custom dataset from {args.custom_data_path}")
        try:
            use_pre_dataset = True
            pre_dataset_file = "/Users/yushi/work/project/tech_tf/sigfin/EDINET-Bench/data/local_ds_2025.pkl"
            if use_pre_dataset:
                with open(
                    pre_dataset_file,
                    "rb",
                ) as inf:
                    ds = pickle.load(inf)
            else:
                creator = CustomDatasetCreator(args.custom_data_path)
                ds = creator.create_dataset(
                    task=args.task,
                    file_pattern=args.custom_data_pattern,
                    max_files=args.max_custom_files,
                )
                logger.info(f"Created custom dataset with {len(ds)} examples")
                with open(
                    pre_dataset_file,
                    "wb",
                ) as f:
                    pickle.dump(ds, f)

        except Exception as e:
            logger.error(f"Failed to create custom dataset: {e}")
            exit()
    else:
        logger.info("Loading dataset from HuggingFace")
        ds = datasets.load_dataset(
            "SakanaAI/EDINET-Bench",
            args.task,
            split="test",
        )

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.num_example:
        ds = ds.select(range(args.num_example))
    if len(ds) == 0:
        logger.error("No data found")
        exit(1)

    model = MODEL_TABLE[args.model](args.model, args.system_prompt)
    result_list = []
    with open(os.path.join("prompt", f"{args.task}.yaml"), "r") as file:
        data = yaml.safe_load(file)
    prompt = data["prompt"]
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_example,
                example,
                model,
                prompt,
                args.sheets,
            )
            for example in ds
        ]
        for future in tqdm(as_completed(futures), total=len(ds)):
            result = future.result()
            logger.info(f"Result: {result}")
            result_list.append(result)

    save_dir = os.path.join(args.output_dir, args.task, args.model)
    os.makedirs(save_dir, exist_ok=True)
    sheets = sorted(args.sheets)
    with open(
        os.path.join(
            save_dir,
            "_".join(sheets) + ".jsonl",
        ),
        "w",
    ) as file:
        for result in result_list:
            file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"saved results to {save_dir}")

    # Display usage statistics and cost
    usage_stats = model.get_usage_stats()
    input_tokens = usage_stats["input_tokens"]
    output_tokens = usage_stats["output_tokens"]
    total_tokens = input_tokens + output_tokens

    logger.info(f"\n{'='*60}")
    logger.info(f"Usage Statistics for {args.model}")
    logger.info(f"{'='*60}")
    logger.info(f"Input tokens:  {input_tokens:,}")
    logger.info(f"Output tokens: {output_tokens:,}")
    logger.info(f"Total tokens:  {total_tokens:,}")

    # Calculate cost based on model
    if "vertex-ai" in args.model:
        # Vertex AI pricing (as of 2025)
        # Claude 3.7 Sonnet: $3.00/M input, $15.00/M output
        input_cost = (input_tokens / 1_000_000) * 3.00
        output_cost = (output_tokens / 1_000_000) * 15.00
        total_cost = input_cost + output_cost

        logger.info(f"\nVertex AI Claude 3.7 Sonnet Pricing:")
        logger.info(f"  Input:  ${input_cost:.4f} (${3.00}/M tokens)")
        logger.info(f"  Output: ${output_cost:.4f} (${15.00}/M tokens)")
        logger.info(f"  Total:  ${total_cost:.4f}")
    elif "claude" in args.model:
        # Anthropic API pricing (as of 2025)
        # Claude 3.7 Sonnet: $3.00/M input, $15.00/M output
        # Claude 3.5 Sonnet: $3.00/M input, $15.00/M output
        # Claude 3.5 Haiku: $0.80/M input, $4.00/M output
        if "haiku" in args.model:
            input_price = 0.80
            output_price = 4.00
        else:
            input_price = 3.00
            output_price = 15.00

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost

        logger.info(f"\nAnthropic API Pricing:")
        logger.info(f"  Input:  ${input_cost:.4f} (${input_price}/M tokens)")
        logger.info(f"  Output: ${output_cost:.4f} (${output_price}/M tokens)")
        logger.info(f"  Total:  ${total_cost:.4f}")
    elif "gpt-4o" in args.model:
        # GPT-4o pricing: $2.50/M input, $10.00/M output
        input_cost = (input_tokens / 1_000_000) * 2.50
        output_cost = (output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + output_cost

        logger.info(f"\nOpenAI GPT-4o Pricing:")
        logger.info(f"  Input:  ${input_cost:.4f} ($2.50/M tokens)")
        logger.info(f"  Output: ${output_cost:.4f} ($10.00/M tokens)")
        logger.info(f"  Total:  ${total_cost:.4f}")

    logger.info(f"{'='*60}\n")
