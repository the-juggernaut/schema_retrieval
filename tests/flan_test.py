from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """
Extract the following data from the text below. Return only valid JSON.

Schema:
{
  "awards": [
    {
      "title": "...",
      "date": "...",
      "awarder": "...",
      "summary": "..."
    }
  ]
}

Text:
He won a scholarship from Time Magazine in 2022 for his groundbreaking work in Quantum Physics.
"""

result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
print(result)
