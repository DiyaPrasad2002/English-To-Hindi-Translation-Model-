# English To Hindi Translation Model and Dataset
The project aims at building an encoder-decoder translation model based on LSTMs within, which enable translations of text from English language to Hindi. The dataset used for the project was created by IITB since 2016, at the Centre For Indian Language Technology. Different derivative corpus of the dataset are available, however, the dataset present on HuggingFace consists of 1,662,110 rows.

## Preprocessing
The multiple steps of preprocessing utilised in the process are :
1. Drop the duplicates
2. Drop all the null values
3. Lowercase all the english words
4. Remove all the quotes and special characters
5. Remove all the numbers and extra spaces from the text
6. Add the start and end tokens to the Hindi sentences
   
## Encoder-Decoder Architecture
Encoder-Decoder models are basically neural network architectures, making use of architectures like RNNs and LSTMs for tasks like machine translation. The encoder part of the architecture takes in the input sequence in one language, generates the context vector. The decoder accepts the context vector as an input and generates the desired output sequence, in the other language.

## Future Possibilities
Whilst I have restricted to the encoder-decoder architecture only, attention layers could be also added in the architecture to make the translator more context specific, thus progressing to more of a transformer-like architecture. This further progresses into construction of the LLM architecture.



