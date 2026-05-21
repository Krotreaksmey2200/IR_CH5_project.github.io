import re
import nltk
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------------------------------------------------
# Load SpaCy English Model
# ---------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------
# Download NLTK Corpus
# ---------------------------------------------------
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------------------------------------------
# Load English Text Corpus
# Using NLTK Gutenberg Corpus
# ---------------------------------------------------
raw_text = gutenberg.raw('wiki_article.txt')

# Use only first 5000 characters for easier display
raw_text = raw_text[:5000]

# ---------------------------------------------------
# Display Original Text
# ---------------------------------------------------
print("\n================ ORIGINAL TEXT ================\n")
print(raw_text)

# ---------------------------------------------------
# Step 1: Normalization and Case Folding
# Convert text to lowercase
# ---------------------------------------------------
text = raw_text.lower()

# ---------------------------------------------------
# Step 2: Remove Special Characters and Punctuation
# Keep only alphabets and spaces
# ---------------------------------------------------
text = re.sub(r'[^a-zA-Z\s]', '', text)

# ---------------------------------------------------
# Step 3: Tokenization
# ---------------------------------------------------
tokens = word_tokenize(text)

# ---------------------------------------------------
# Step 4: Stop-word Removal
# ---------------------------------------------------
stop_words = set(stopwords.words('english'))

filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]

# ---------------------------------------------------
# Step 5: Stemming
# ---------------------------------------------------
stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_tokens
]

# ---------------------------------------------------
# Step 6: Lemmatization
# ---------------------------------------------------
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_tokens
]

# ---------------------------------------------------
# Create Cleaned Text
# ---------------------------------------------------
cleaned_text = " ".join(lemmatized_words)

# ---------------------------------------------------
# Display Cleaned Text
# ---------------------------------------------------
print("\n================ CLEANED TEXT ================\n")
print(cleaned_text)

# ---------------------------------------------------
# Display First 20 Tokens
# ---------------------------------------------------
print("\n================ FIRST 20 TOKENS ================\n")
print(tokens[:20])

# ---------------------------------------------------
# Word Counts
# ---------------------------------------------------
original_word_count = len(word_tokenize(raw_text))
cleaned_word_count = len(lemmatized_words)

print("\n================ WORD COUNTS ================\n")
print(f"Original Word Count : {original_word_count}")
print(f"Cleaned Word Count  : {cleaned_word_count}")

# ---------------------------------------------------
# Compare Stemming vs Lemmatization
# ---------------------------------------------------
print("\n================ STEMMING vs LEMMATIZATION ================\n")

for i in range(20):
    print(
        f"Original: {filtered_tokens[i]:15}"
        f"Stemmed: {stemmed_words[i]:15}"
        f"Lemmatized: {lemmatized_words[i]}"
    )

# ---------------------------------------------------
# Top 20 Frequent Words
# ---------------------------------------------------
word_freq = Counter(lemmatized_words)

top_20 = word_freq.most_common(20)

print("\n================ TOP 20 FREQUENT WORDS ================\n")

for word, count in top_20:
    print(f"{word:<15} {count}")

# ---------------------------------------------------
# Bar Chart
# ---------------------------------------------------
words = [item[0] for item in top_20]
counts = [item[1] for item in top_20]

plt.figure(figsize=(14, 7))

plt.bar(words, counts)

plt.title("Top 20 Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ---------------------------------------------------
# Compare Original vs Cleaned Text
# ---------------------------------------------------
print("\n================ ORIGINAL vs CLEANED ================\n")

print("----- ORIGINAL TEXT SAMPLE -----\n")
print(raw_text[:1000])

print("\n----- CLEANED TEXT SAMPLE -----\n")
print(cleaned_text[:1000])

# ---------------------------------------------------
# SpaCy Example (Optional Additional Processing)
# ---------------------------------------------------
print("\n================ SPACY TOKENIZATION SAMPLE ================\n")

doc = nlp(raw_text[:500])

for token in doc[:20]:
    print(token.text, "->", token.lemma_)