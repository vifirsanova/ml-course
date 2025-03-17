### **Базовый уровень: Генерация текста с использованием LSTM**

0. **Сбор данных**: используйте готовый или соберите свой корпус в формате plain text для генерации текстов

1. Генерация текста на основе небольшого датасета
   - Предварительный анализ: чистка текста
   - Обучение модели. Пример:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import LSTM, Dense, Embedding

     model = Sequential([
         Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_len),
         LSTM(128, return_sequences=True),
         LSTM(128),
         Dense(vocab_size, activation='softmax')
     ])
     model.compile(loss='categorical_crossentropy', optimizer='adam')
     model.fit(X_train, y_train, epochs=20, batch_size=128)
     ```
   - Генерация текста. Пример:
     ```python
     def generate_text(seed_text, next_words, max_sequence_len):
         for _ in range(next_words):
             token_list = tokenizer.texts_to_sequences([seed_text])[0]
             token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
             predicted = model.predict(token_list, verbose=0)
             predicted_word = tokenizer.index_word[predicted.argmax()]
             seed_text += " " + predicted_word
         return seed_text
     ```

---

### **Продвинутый уровень: Машинный перевод с использованием LSTM**

0. **Сбор данных**: используйте готовый или соберите свой параллельный корпус (например, **OpenSubtitles**)
     
1. Реализация **seq2seq** модели

   - Построение модели **LSTM**, **GRU**, другая рекуррентная архитектура на выбор
   - Релизуем классы кодера и декодера

2. Опционально: добавление механизма внимания (**attention**)
   
   - Пример **tf.Attention**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
   
3. Оценка качества обучения с помощью **perplexity**, **BLEU score**, других метрик оценки
