from anthropic import AnthropicVertex

model_name: str = "claude-3-7-sonnet@20250219"
system_prompt: str = "You are a helpful assistant."
location: str = "us-east5"
project_id: str = "tech-verification-265409"

client = AnthropicVertex(region=location, project_id=project_id)
prompt = "トヨタの2022年の利益は前年に比べて増加しましたか？"
prompt = "日本史上最高気温は?webを検索してください。"
response = client.messages.create(
    model=model_name,
    system=system_prompt,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=10000,
    temperature=0.0,
)

print(response.content[0].text)
