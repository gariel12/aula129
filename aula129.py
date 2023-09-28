import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Baixe as stopwords e o lemmatizer (caso ainda não tenha feito isso)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Frase de exemplo
texto = "O pré-processamento de texto é uma etapa importante na análise de dados textuais. Ele envolve a tokenização, remoção de stopwords e lematização."

# Tokenização usando o Tokenizer do TensorFlow
tokenizer = tf.keras.layers.TextVectorization(max_tokens=100, output_mode='int')
tokenizer.adapt([texto])
tokens = tokenizer(texto)

# Remoção de stopwords: elimina palavras comuns que geralmente não contribuem para a análise
stop_words = set(stopwords.words('portuguese'))  # Você pode escolher outro idioma se necessário
tokens_sem_stopwords = [word for word in tokens.numpy()[0] if word.lower() not in stop_words]

# Lematização: reduz as palavras às suas formas básicas (lemmas)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_sem_stopwords]

# Imprime o resultado
print("Texto original:")
print(texto)

print("\nTokens após pré-processamento:")
print(lemmatized_tokens)
