from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import nltk

# Download NLTK tokenizers (new requirement)
#nltk.download('punkt')
#nltk.download('punkt_tab')


finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# build model and pipeline
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


with open("article.txt", "r", encoding="utf-8") as f:
    article_text = f.read()

print(f"Article text tokens: {len(tokenizer.encode(article_text, add_special_tokens=True))} tokens")

sentences = nltk.sent_tokenize(article_text)
results = nlp(sentences)
#print(results)

result = nlp(article_text)
print(result)
