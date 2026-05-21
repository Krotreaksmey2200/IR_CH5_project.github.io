import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import re
import nltk
import spacy
import matplotlib.pyplot as plt
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# LOAD SPACY MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")


class NLPApp:

    def __init__(self, root):
        self.root = root
        self.root.title("English NLP Text Processing System")
        self.root.geometry("1100x800")

        self.text = ""

        # ---------------- BUTTONS ----------------
        tk.Button(root, text="1. Load Corpus (TXT)", command=self.load_file, bg="green", fg="white").pack()

        tk.Button(root, text="2. Process Text", command=self.process, bg="blue", fg="white").pack()

        tk.Button(root, text="3. Show Bar Chart", command=self.plot_chart, bg="purple", fg="white").pack()

        # ---------------- OUTPUT BOXES ----------------
        tk.Label(root, text="Original Text").pack()
        self.original_box = scrolledtext.ScrolledText(root, height=6)
        self.original_box.pack()

        tk.Label(root, text="Cleaned Text").pack()
        self.cleaned_box = scrolledtext.ScrolledText(root, height=6)
        self.cleaned_box.pack()

        tk.Label(root, text="First 20 Tokens").pack()
        self.tokens_box = scrolledtext.ScrolledText(root, height=3)
        self.tokens_box.pack()

        tk.Label(root, text="Word Counts").pack()
        self.count_box = scrolledtext.ScrolledText(root, height=3)
        self.count_box.pack()

        tk.Label(root, text="Stemming vs Lemmatization").pack()
        self.compare_box = scrolledtext.ScrolledText(root, height=6)
        self.compare_box.pack()

        tk.Label(root, text="Top 20 Words").pack()
        self.top_box = scrolledtext.ScrolledText(root, height=6)
        self.top_box.pack()

        self.freq_data = []

    # ---------------- LOAD FILE ----------------
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()

            self.original_box.delete("1.0", tk.END)
            self.original_box.insert(tk.END, self.text[:3000])

    # ---------------- PROCESS TEXT ----------------
    def process(self):

        if not self.text:
            messagebox.showerror("Error", "Please load a corpus first")
            return

        # ---------------- NORMALIZATION ----------------
        text = self.text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # ---------------- TOKENIZATION ----------------
        tokens = word_tokenize(text)

        # ---------------- STOPWORDS ----------------
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w not in stop_words]

        # ---------------- STEMMING ----------------
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(w) for w in filtered]

        # ---------------- LEMMATIZATION ----------------
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

        cleaned_text = " ".join(lemmatized)

        # ---------------- DISPLAY ----------------
        self.cleaned_box.delete("1.0", tk.END)
        self.cleaned_box.insert(tk.END, cleaned_text[:3000])

        self.tokens_box.delete("1.0", tk.END)
        self.tokens_box.insert(tk.END, str(tokens[:20]))

        self.count_box.delete("1.0", tk.END)
        self.count_box.insert(tk.END,
                              f"Original Tokens: {len(word_tokenize(self.text))}\n"
                              f"After Cleaning: {len(lemmatized)}")

        self.compare_box.delete("1.0", tk.END)
        for i in range(min(20, len(filtered))):
            self.compare_box.insert(
                tk.END,
                f"{filtered[i]} | Stem: {stemmed[i]} | Lemma: {lemmatized[i]}\n"
            )

        # ---------------- TOP WORDS ----------------
        freq = Counter(lemmatized)
        self.freq_data = freq.most_common(20)

        self.top_box.delete("1.0", tk.END)
        for word, count in self.freq_data:
            self.top_box.insert(tk.END, f"{word}: {count}\n")

    # ---------------- BAR CHART ----------------
    def plot_chart(self):

        if not self.freq_data:
            messagebox.showerror("Error", "Process text first")
            return

        words = [w[0] for w in self.freq_data]
        counts = [c[1] for c in self.freq_data]

        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.title("Top 20 Frequent Words")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# ---------------- RUN APP ----------------
root = tk.Tk()
app = NLPApp(root)
root.mainloop()
