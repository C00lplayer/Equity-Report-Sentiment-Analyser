from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")
result = pipe("The company's revenue failed to meet expectations this quarter.")
print(result)
