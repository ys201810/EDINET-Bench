import anthropic
import os
import backoff
from dotenv import load_dotenv
import re
import json
import weave
import openai
import pandas as pd
import numpy as np

load_dotenv()


@weave.op()
@backoff.on_exception(
    backoff.expo, (anthropic.RateLimitError, anthropic.APITimeoutError), max_tries=3
)
def get_response_from_llm(
    user_text: str,
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    messages: list | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[str, list]:
    if messages is None:
        messages = []
    messages = messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                }
            ],
        }
    ]
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    content = response.content[0].text
    messages = messages + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    ]
    return content, messages


# openai api


@weave.op()
@backoff.on_exception(
    backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=3
)
def get_response_from_gpt(
    user_text: str,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    messages: list | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[str, list]:
    if messages is None:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
    messages = messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                }
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=0,
    )
    content = response.choices[0].message.content
    messages = messages + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    ]
    return content, messages


def extract_json_between_markers(llm_output: str | None) -> dict | None:
    # Handle None input
    if llm_output is None:
        print("No output from LLM")
        return None

    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create differential features from time-series financial data.
    
    Args:
        df: DataFrame with columns like 'metric_Prior4Year', 'metric_Prior3Year', etc.
    
    Returns:
        DataFrame with additional differential features
    """
    df_diff = df.copy()
    
    # Find all unique metric names (excluding year suffix)
    metric_names = set()
    for col in df.columns:
        if col != 'label':
            # Extract metric name by removing year suffix
            parts = col.split('_')
            if len(parts) >= 2 and parts[-1] in ['Prior4Year', 'Prior3Year', 'Prior2Year', 'Prior1Year', 'CurrentYear']:
                metric_name = '_'.join(parts[:-1])
                metric_names.add(metric_name)
    
    # Create differential features for each metric
    for metric in metric_names:
        years = ['Prior4Year', 'Prior3Year', 'Prior2Year', 'Prior1Year', 'CurrentYear']
        cols = [f"{metric}_{year}" for year in years]
        
        # Check if all required columns exist
        existing_cols = [col for col in cols if col in df.columns]
        if len(existing_cols) < 2:
            continue
            
        # Sort columns by year (oldest to newest)
        existing_cols = sorted(existing_cols, key=lambda x: years.index(x.split('_')[-1]))
        
        # Create year-over-year differences
        for i in range(1, len(existing_cols)):
            prev_col = existing_cols[i-1]
            curr_col = existing_cols[i]
            diff_col = f"{metric}_diff_{years[years.index(curr_col.split('_')[-1])]}_vs_{years[years.index(prev_col.split('_')[-1])]}"
            
            # Calculate difference (current - previous)
            df_diff[diff_col] = df[curr_col] - df[prev_col]
            
        # Create growth rates (percentage change)
        for i in range(1, len(existing_cols)):
            prev_col = existing_cols[i-1]
            curr_col = existing_cols[i]
            growth_col = f"{metric}_growth_{years[years.index(curr_col.split('_')[-1])]}_vs_{years[years.index(prev_col.split('_')[-1])]}"
            
            # Calculate growth rate with division by zero protection
            with np.errstate(divide='ignore', invalid='ignore'):
                growth_rate = (df[curr_col] - df[prev_col]) / df[prev_col].abs()
                # Replace inf and -inf with NaN
                growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan)
                df_diff[growth_col] = growth_rate
    
    return df_diff


def test_extract_json_between_markers():
    llm_output = """
    THOUGHT:
    <THOUGHT>

    REVIEW JSON:
    ```json
    {
        "prediction": 1,
        "reasoning": "The financial report contains fraudulent accounting practices."
    }
    ```
    """
    result = extract_json_between_markers(llm_output)
    assert result == {
        "prediction": 1,
        "reasoning": "The financial report contains fraudulent accounting practices.",
    }


if __name__ == "__main__":
    print("Asking another question")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response, messages = get_response_from_llm(
        "こんにちは, 今何をしていますか?",
        client,
        "claude-3-5-sonnet-20241022",
        "You are an expert in financial fraud detection.",
    )
    print(response)
    print(messages)
    print("Asking another question")
    response, messages = get_response_from_llm(
        "日本銀行とは?",
        client,
        "claude-3-5-sonnet-20241022",
        "You are an expert in financial fraud detection.",
        messages=messages,
    )
    print(response)
    print(messages)

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response, messages = get_response_from_gpt(
        "こんにちは, 今何をしていますか?",
        client,
        "gpt-3.5-turbo",
        "You are an expert in financial fraud detection.",
    )
    print(response)
    print(messages)
