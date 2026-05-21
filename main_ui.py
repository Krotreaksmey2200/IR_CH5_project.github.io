import sys
import re
import nltk
import spacy
import matplotlib.pyplot as plt
from collections import Counter

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")


class NLPApp(QtWidgets.QMainWindow):

    def __init__(self):
        super(NLPApp, self).__init__()
        uic.loadUi("nlp_gui.ui", self)

        self.text = ""
        self.freq_data = []

        # BUTTON EVENTS
        self.btnLoad.clicked.connect(self.load_file)
        self.btnProcess.clicked.connect(self.process_text)
        self.btnChart.clicked.connect(self.show_chart)

    # ---------------- LOAD FILE ----------------
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open TXT File", "", "Text Files (*.txt)")

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()

            self.txtOriginal.setText(self.text[:3000])

    # ---------------- PROCESS TEXT ----------------
    def process_text(self):

        if not self.text:
            QMessageBox.warning(self, "Error", "Please load file first")
            return

        text = self.text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w not in stop_words]

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        stemmed = [stemmer.stem(w) for w in filtered]
        lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

        cleaned = " ".join(lemmatized)

        # ---------------- DISPLAY ----------------
        self.txtClean.setText(cleaned[:3000])
        self.txtTokens.setText(str(tokens[:20]))

        self.txtCount.setText(
            f"Original: {len(word_tokenize(self.text))}\n"
            f"After Cleaning: {len(lemmatized)}"
        )

        self.txtCompare.clear()
        for i in range(min(20, len(filtered))):
            self.txtCompare.append(
                f"{filtered[i]} | Stem: {stemmed[i]} | Lemma: {lemmatized[i]}"
            )

        freq = Counter(lemmatized)
        self.freq_data = freq.most_common(20)

        self.txtTop.clear()
        for w, c in self.freq_data:
            self.txtTop.append(f"{w}: {c}")

    # ---------------- BAR CHART ----------------
    def show_chart(self):

        if not self.freq_data:
            QMessageBox.warning(self, "Error", "Process text first")
            return

        words = [w[0] for w in self.freq_data]
        counts = [w[1] for w in self.freq_data]

        plt.figure(figsize=(10,5))
        plt.bar(words, counts)
        plt.xticks(rotation=45)
        plt.title("Top 20 Words")
        plt.show()


# ---------------- RUN APP ----------------
app = QtWidgets.QApplication(sys.argv)
window = NLPApp()
window.show()
sys.exit(app.exec_())