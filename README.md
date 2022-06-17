# streaming_aai_codes

This is the code repository for the paper - "Streaming model for Acoustic to Articulatory Inversion with transformer networks" (Sathvik Udupa, Aravind Illa and Prasanta Ghosh), accepted for presentation at the Interspeech 2022 conference. Paper link & citation to be added soon.

The code is present mostly as it is from experiments, for my queries, feel free to reach out at sathvikudupa66@gmail.com

Summary:
We perform full sequence training with transformer and allow for chunk level autoregressive transfer for the sequence - to -sequence task of AAI. We obtain significant perfornace improvement with a range of masking operations inside the tranformer self attention, while it is training.
Note that the proposed approach can be applied to any streaming sequence-to-sequence tasks where the input and output have the same length.

