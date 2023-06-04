import argparse


def pars_args():
    parser = argparse.ArgumentParser(description="HeterSumGraph Model")

    # Where to find data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="G:\dars\\arshad\\tez\projects\HSG\datasets\cnndm",
        help="The dataset directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="G:\dars\\arshad\\tez\projects\HSG\HeterSumGraph\cache\CNNDM",
        help="The processed dataset directory",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="G:\dars\\arshad\\tez\projects\HSG\embeddings\glove.42B.300d.txt",
        help="Path expression to external word embedding.",
    )

    # Important settings
    parser.add_argument(
        "--model", type=str, default="HSG", help="model structure[HSG|HDSG]"
    )
    ###
    # ! Early stopping: Form of regularization used to avoid overfitting on the training dataset.
    ###
    parser.add_argument(
        "--restore_model",
        type=str,
        default="None",
        help="Restore model for further training. [bestmodel/bestFmodel/earlystop/None]",
    )

    # Where to save output
    parser.add_argument(
        "--save_root", type=str, default="save/", help="Root directory for all model."
    )
    parser.add_argument(
        "--log_root", type=str, default="log/", help="Root directory for all logging."
    )

    # Hyperparameters
    parser.add_argument(
        "--seed", type=int, default=666, help="set the random seed [default: 666]"
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU ID to use. [default: 0]"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="GPU or CPU [default: False]"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Size of vocabulary. [default: 50000]",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="Number of epochs [default: 20]"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Mini batch size [default: 32]"
    )
    parser.add_argument(
        "--n_iter", type=int, default=1, help="iteration hop [default: 1]"
    )

    parser.add_argument(
        "--word_embedding",
        action="store_true",
        default=False,
        help="whether to use Word embedding [default: True]",
    )
    parser.add_argument(
        "--word_emb_dim",
        type=int,
        default=300,
        help="Word embedding size [default: 300]",
    )
    parser.add_argument(
        "--embed_train",
        action="store_true",
        default=False,
        help="whether to train Word embedding [default: False]",
    )
    parser.add_argument(
        "--feat_embed_size",
        type=int,
        default=50,
        help="feature embedding size [default: 50]",
    )
    parser.add_argument(
        "--n_layers", type=int, default=1, help="Number of GAT layers [default: 1]"
    )
    parser.add_argument(
        "--lstm_hidden_state",
        type=int,
        default=128,
        help="size of lstm hidden state [default: 128]",
    )
    parser.add_argument(
        "--lstm_layers", type=int, default=2, help="Number of lstm layers [default: 2]"
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="whether to use bidirectional LSTM [default: True]",
    )
    parser.add_argument(
        "--n_feature_size",
        type=int,
        default=128,
        help="size of node feature [default: 128]",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size [default: 64]"
    )
    parser.add_argument(
        "--ffn_inner_hidden_size",
        type=int,
        default=512,
        help="PositionwiseFeedForward inner hidden size [default: 512]",
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="multihead attention number [default: 8]"
    )
    parser.add_argument(
        "--recurrent_dropout_prob",
        type=float,
        default=0.1,
        help="recurrent dropout prob [default: 0.1]",
    )
    parser.add_argument(
        "--atten_dropout_prob",
        type=float,
        default=0.1,
        help="attention dropout prob [default: 0.1]",
    )
    parser.add_argument(
        "--ffn_dropout_prob",
        type=float,
        default=0.1,
        help="PositionwiseFeedForward dropout prob [default: 0.1]",
    )
    parser.add_argument(
        "--use_orthnormal_init",
        action="store_true",
        default=True,
        help="use orthnormal init for lstm [default: True]",
    )
    parser.add_argument(
        "--sent_max_len",
        type=int,
        default=100,
        help="max length of sentences (max source text sentence tokens)",
    )
    parser.add_argument(
        "--doc_max_timesteps",
        type=int,
        default=50,
        help="max length of documents (max timesteps of documents)",
    )

    # Training
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument(
        "--lr_descent", action="store_true", default=False, help="learning rate descent"
    )
    parser.add_argument(
        "--grad_clip", action="store_true", default=False, help="for gradient clipping"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="for gradient clipping max gradient normalization",
    )

    parser.add_argument("-m", type=int, default=3, help="decode summary length")

    args = parser.parse_args()

    return args


# ****************************************************************

# ---------------------------- About Data ----------------------------

# train.w2s.tfidf.jsonl => 287084
# train.label.jsonl => 287084

# test.w2s.tfidf.jsonl => 11489
# test.label.jsonl => 11489

# val.w2s.tfidf.jsonl => 13367
# val.label.jsonl => 13367

#    => 1917494

# ****************************************************************

# ---------------------------- Dataset ----------------------------
# Data number 1 => train
"""
(Graph(num_nodes=156, num_edges=1054,
      ndata_schemes=
      {
            'unit': Scheme(shape=(), dtype=torch.float32), 
            'id': Scheme(shape=(), dtype=torch.int64), 
            'dtype': Scheme(shape=(), dtype=torch.float32), 
            'words': Scheme(shape=(100,), dtype=torch.int64), 
            'position': Scheme(shape=(1,), dtype=torch.int64), 
            'label': Scheme(shape=(50,), dtype=torch.int64)
      }
      edata_schemes=
      {
            'tffrac': Scheme(shape=(), dtype=torch.int64), 
            'dtype': Scheme(shape=(), dtype=torch.float32)
      }), 1)
"""
# Data number 1 => validation
"""
(Graph(num_nodes=92, num_edges=778,
      ndata_schemes=
      {
        'unit': Scheme(shape=(), dtype=torch.float32), 
        'id': Scheme(shape=(), dtype=torch.int64), 
        'dtype': Scheme(shape=(), dtype=torch.float32), 
        'words': Scheme(shape=(100,), dtype=torch.int64), 
        'position': Scheme(shape=(1,), dtype=torch.int64), 
        'label': Scheme(shape=(50,), dtype=torch.int64)
      }
      edata_schemes=
      {
        'tffrac': Scheme(shape=(), dtype=torch.int64), 
        'dtype': Scheme(shape=(), dtype=torch.float32)}), 1)
"""
# Data Loader 1 => validation
"""
    (Graph(num_nodes=6882, num_edges=86904,
      ndata_schemes=
      {
        'unit': Scheme(shape=(), dtype=torch.float32), 
        'id': Scheme(shape=(), dtype=torch.int64), 
        'dtype': Scheme(shape=(), dtype=torch.float32), 
        'words': Scheme(shape=(100,), dtype=torch.int64), 
        'position': Scheme(shape=(1,), dtype=torch.int64), 
        'label': Scheme(shape=(50,), dtype=torch.int64)}
      edata_schemes=
      {
        'tffrac': Scheme(shape=(), dtype=torch.int64), 
        'dtype': Scheme(shape=(), dtype=torch.float32)}), [3, 4, 12, 26, 28, 31, 15, 30, 23, 11, 22, 13, 19, 0, 9, 25, 14, 18, 7, 27, 10, 21, 2, 8, 1, 17, 24, 29, 16, 20, 5, 6])
"""
# ****************************************************************
