# Hindi-to-English-NMT
Hindi to English Neural Machine Translation.

I have implemented Sequence to Sequence based models with LSTM encoder and decoder, Attention model with Bi-LSTM encoder and Attention model with Bi-GRU encoder. Different
optimizers like Adam, RMSprop and AdamW are used and AdamW worked best for NMT from Hindi to English. A train set with 102323 Hindi and English sentence pairs is provided
to train the models. The dataset has been curated from the following publicly available sources: 
<a href="https://opus.nlpl.eu/ and https://www.opensubtitles.org/en/search">
</a> 
The models are evaluated on the basis of Bleu-4 score, Bleu-4 macro score and Meteor score. 
## Note: Using Google Colab with GPU runtime is recommended. 
Link to the files needed: https://drive.google.com/drive/u/0/folders/1w5Wn1YSpJHNgecwCmwXrbmpuAziuyQSx


Different Models used for carrying out the NMT are:
1. Seq2Seq (LSTM) with uni-directional bi-layer (Encoder and Decoder), with single context vector fed to initial time-step of Decoder.

<a href="https://colab.research.google.com/drive/1dzWz6Y9rX7-gPzcIMOjVT5rO4sIHOpFX?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

2. Seq2Seq (GRU) with uni-directional uni-layer (Encoder and Decoder) with single context vector fed to each time-step of Decoder.

<a href="https://colab.research.google.com/drive/1X7RmqseeoDcnY16fW2Su07MU8A-U_LBn?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

3. Seq2Seq (GRU) with with bi-directional uni-layer Encoder and Uni-directional uni-layer Decoder with Attention Mechanism

<a href="https://colab.research.google.com/drive/1rChM3nAflAQ3mMOTTne4R8vGjwK4sIss?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>