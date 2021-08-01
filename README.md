# Hindi-to-English-NMT
Hindi to English Neural Machine Translation.

I have implemented Sequence to Sequence based models with LSTM encoder and decoder, Attention model with Bi-LSTM encoder and Attention model with Bi-GRU encoder. Different
optimizers like Adam, RMSprop and AdamW are used and AdamW worked best for NMT from Hindi to English. A train set with 102323 Hindi and English sentence pairs is provided
to train the models. The dataset has been curated from the following publicly available sources: 
<a href="https://opus.nlpl.eu/ and https://www.opensubtitles.org/en/search">
</a> 
The models are evaluated on the basis of Bleu-4 score, Bleu-4 macro score and Meteor score. 
## Note: Using Google Colab with GPU runtime is recommended. 
All the datasets and models can be accessed here: https://drive.google.com/drive/folders/12Blaa-pyPm9bCMpvCRWQRIGNfKRWsNG7?usp=sharing


Different Models used for carrying out the NMT are:
## 1. Seq2Seq (LSTM) with uni-directional bi-layer (Encoder and Decoder), with single context vector fed to initial time-step of Decoder.
This notebook implements LSTM based uni-directional bi-layer Encoder and Decoder and uses torchtext to do all of the heavy lifting with regards to text processing.
<a href="https://colab.research.google.com/drive/1DLqPtqfFTCcSlh0rkD5uM6gaL69Kbd0o?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 2. Seq2Seq Model with uni-directional bi-layer LSTM Encoder and Decoder, with single context vector fed to initial time-step of Decoder.
This model is also an implementation of the model described above however in this phase all the functionalities of torchtext are replaced using light weight functions provided by python. 
<a href="https://colab.research.google.com/drive/1Sto43hDfcJFrb1C7fWpRxVmyLnUKUqv4?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 3. Seq2Seq model with with bi-directional uni-layer GRU Encoder and Uni-directional uni-layer GRU Decoder with Attention Mechanism. 
This model is based on the idea proposed by Bahdanau et al. in the paper Neural Machine Translation by Jointly Learning to Align and Translate. The model uses attention mechanism, where the decoding process is guided by an attention vector which represents which words in the source we should pay the most attention to
in order to correctly decode the next word.
<a href="https://colab.research.google.com/drive/1Sto43hDfcJFrb1C7fWpRxVmyLnUKUqv4?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
