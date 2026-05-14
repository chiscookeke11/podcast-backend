import os
import anthropic

client = anthropic.Anthropic(api_key="YOUR_KEY")

print(client.models.list())
