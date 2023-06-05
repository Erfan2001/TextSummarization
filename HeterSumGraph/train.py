import datetime
import os
import shutil
import time
import random
import numpy as np
import torch
from rouge import Rouge
import dgl
from config import pars_args
from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *

# Debug Flag
_DEBUG_FLAG_ = False


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, hps):
    #region
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """
    # Path of Training model => HeterSumGraph\results\log\train
    train_dir = os.path.join(hps.save_root, "train")
    # --------- Restoring Model (PreTrained Models) ---------
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        # --------- Creating New File for Model (If Not Exists) ---------
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)
    # --------- Start Training ---------
    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir)
    # --------- Interrupt (Early Stop) ---------
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))
    #endregion

def run_training(model, train_loader, valid_loader, valset, hps, train_dir):
    #region
    """  Repeatedly runs training iterations, logging loss to screen and log files

        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return:
    """
    logger.info("[INFO] Starting run_training")

    # --------- Define Optimizer ---------
    """
        learning rate: 0.0005
        Adam Optimizer: Extended version of stochastic gradient descent
        Model Parameters (Example):
            1) Data:
                tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                [ 1.0435,  0.0049, -0.9158,  ..., -1.4600, -0.7835, -0.8562],
                [ 0.0871, -0.6995,  0.6882,  ..., -0.0618,  1.2191, -1.2322],
                ...,
                [-0.5717, -1.1029,  0.3211,  ..., -0.0190, -0.2724, -1.3183],
                [ 0.1839, -1.6656, -1.0831,  ...,  1.8187, -1.9443,  0.6668],
                [-1.6763, -0.8578,  0.3160,  ..., -0.3354,  0.3789, -1.7529]],
                device='cuda:0')
            2) Shape: torch.Size([50000, 300])
    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    # --------- Loss Function ---------
    # !!Explain: Cross entropy loss between input logits and target
    """
        Some Loss Functions:
            1) Mean Absolute Error (L1 Loss Function | nn.L1Loss()): Sum of absolute differences between actual values and predicted values.
            2) Mean Squared Error (MSE | nn.MSELoss()):  average of the squared differences between actual values and predicted values.
            3) Negative Log-Likelihood(NLL | nn.NLLLoss()): for models with the softmax function as an output activation layer.
            4) Cross-Entropy (CE | nn.CrossEntropyLoss()): difference between two probability distributions for a provided set of occurrences or random variables.
            5) Hinge Embedding (HE | nn.HingeEmbeddingLoss()): Target values are between {1, -1}, which makes it good for binary classification tasks.
            6) Margin Ranking (MR | nn.MarginRankingLoss()): computes a criterion to predict the relative distances between inputs.(predict directly from a given set of inputs)
            7) Triplet Margin (TM | nn.TripletMarginLoss(margin=1.0, p=2)): computes a criterion for measuring the triplet loss in models.
            8) Kullback-Leibler Divergence (KL | nn.KLDivLoss(reduction = 'batchmean')): compute the amount of lost information (expressed in bits) in case the predicted probability distribution is utilized to estimate the expected target probability distribution.
    """
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    # --------- Start Training Epoch ---------
    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        # --------- Start Training Iterator ---------
        """
            Graph(Example):
                Graph(num_nodes=3989, num_edges=52494,
                    ndata_schemes={
                        'unit': Scheme(shape=(), dtype=torch.float32), 
                        'id': Scheme(shape=(), dtype=torch.int64), 
                        'dtype': Scheme(shape=(), dtype=torch.float32), 
                        'words': Scheme(shape=(100,), dtype=torch.int64), 
                        'position': Scheme(shape=(1,), dtype=torch.int64), 
                        'label': Scheme(shape=(50,), dtype=torch.int64)}
                    edata_schemes={
                        'tffrac': Scheme(shape=(), dtype=torch.int64), 
                        'dtype': Scheme(shape=(), dtype=torch.float32)
                                  })
            index(Example):
                [10, 9, 11, 12, 5, 7, 3, 2, 4, 8, 13, 6, 14, 1, 0, 15]
            outputs(Example):
             1) Data: 
                tensor([[ 2.9612e-01,  3.6615e-01],
                        [ 3.4196e-01,  2.6532e-01],
                        [ 2.1960e-01, -1.5018e-01],
                        [ 2.0823e-01, -3.4699e-02],
                        [ 3.6691e-01, -2.8408e-01],
                        [ 1.3301e-02, -2.6481e-01],
                        [ 1.3135e-01,  3.5627e-02],
                        ...
                        [ 9.4292e-01,  3.4942e-01],
                        [ 9.7441e-01, -2.1102e-01],
                        [ 8.1181e-01, -6.9850e-01],
                        [ 1.0028e+00, -1.1024e-01]], 
                device='cuda:0', grad_fn=<AddmmBackward0>)
            2) Size:
                torch.Size([551, 2]) 
        """
        for i, (G, index) in enumerate(train_loader):
            iter_start_time = time.time()
            model.train()
            # Save Model to GPU/CPU
            G = G.to(hps.device)
            # Forward Section of Model
            # !!Explain: [n_snodes, 2]
            outputs = model.forward(G)
            # Separate Node Ids => Size(torch.Size([551]))
            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            # All labels of nodes (n_nodes)
            label = G.ndata["label"][snode_id].sum(-1)
            # Calculate loss function for each node
            # !!Explain: [n_nodes, 1] 
            """
                Example: 
                tensor([[0.7288],
                        [0.6556],
                        [0.5253],
                        [0.5790],
                        [0.4197],
                        [0.5637],
                        [0.6464],
                , device='cuda:0', grad_fn=<IndexSelectBackward0>)
                Size: torch.Size([551, 1])
            """
            G.nodes[snode_id].data["loss"] = criterion(outputs, label.to(hps.device)).unsqueeze(-1)
            # Sums all the values of node field feature in graph
            # !!Explain: [batch_size, 1]
            loss = dgl.sum_nodes(G, "loss")
            loss = loss.mean()
            # If loss values is infinite
            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                raise Exception("train Loss is not finite. Stopping.")
            # Zero all the gradients of the variable (it will update the learnable weights of the model.)
            optimizer.zero_grad()
            # Backward Section of Model
            # !!Explain: Backward tensor(18.4536, device='cuda:0', grad_fn=<MeanBackward0>)
            loss.backward()
            #  Gradient Clipping (Mitigate the problem of exploding gradients)
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)
            # Update based on the current gradient
            optimizer.step()
            # Calculate train and epoch loss
            train_loss += float(loss.data)
            epoch_loss += float(loss.data)
            # Show values every 100 iterations
            if i % 100 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                            .format(i, (time.time() - iter_start_time), float(train_loss / 100)))
                train_loss = 0.0
        # Descent learning rate (Update!!!)
        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)
        # Average loss of each epoch
        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))
        # Save Best Model (Based on Average Loss of epoch)
        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        # If training loss does not decrease(1 time) => stop running and save current model
        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)
        # --------- Start Evaluation Iterator ---------
        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F,
                                                              non_descent_cnt, saveNo)
        # If evaluation loss does not stop(3 times) => stop running and save current model
        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            return
    #endregion

def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo):
    #region
    """
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss
        seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return:
    """
    logger.info("[INFO] Starting eval for this model ...")
    # Make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    # Change to evaluation mode
    model.eval()

    iter_start_time = time.time()
    # Tester ???
    """
        Graph: 
            Graph(num_nodes=1872, num_edges=21680,
                ndata_schemes={
                    'unit': Scheme(shape=(), dtype=torch.float32), 
                    'id': Scheme(shape=(), dtype=torch.int64), 
                    'dtype': Scheme(shape=(), dtype=torch.float32), 
                    'words': Scheme(shape=(100,), dtype=torch.int64), 
                    'position': Scheme(shape=(1,), dtype=torch.int64), 
                    'label': Scheme(shape=(50,), dtype=torch.int64)}
                edata_schemes={
                    'tffrac': Scheme(shape=(), dtype=torch.int64), 
                    'dtype': Scheme(shape=(), dtype=torch.float32)})
        Index: [3, 4, 0, 9, 7, 2, 8, 1, 5, 6]
    """
    with torch.no_grad():
        tester = SLTester(model, hps.m)
        for i, (G, index) in enumerate(loader):
            G = G.to(hps.device)
            tester.evaluation(G, index, valset)
    # Average validation loss
    running_avg_loss = tester.running_avg_loss

    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return
    # Define Rouge measurement
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                      float(running_avg_loss)))
    log_score(scores_all=scores_all)
    # Calculate metrics
    tester.getMetric()
    # F1 score
    F = tester.labelMetric
    # --------- Saving best model (find lower loss)  => Considering loss value ---------
    """
        1) non_descent_cnt: keep track of the number of epochs or iterations during the training process where the optimization algorithm
        did not result in a decrease in the loss function.
    """
    if best_loss is None or running_avg_loss < best_loss:
        # This is where checkpoints of best models are saved
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (saveNo % 3))
        # Best Loss != None
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        # Best Loss == None
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        # Save and Update average loss value
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    # Increase the value of "non_descent_cnt" => Better model was not found.
    else:
        non_descent_cnt += 1
    # --------- Find the best F1 score ---------
    if best_F is None or best_F < F:
        # This is where checkpoints of best models are saved
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel')
        # Best F1 score != None
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F),
                        float(best_F), bestmodel_save_path)
        # Best F1 score == None
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s', float(F),
                        bestmodel_save_path)
        # Save model and Update the best F1 score value
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo
    #endregion

def main():
    args = pars_args()
    # --------- Set Seeds (Python - Numpy - Torch) ---------
    # !!Explain: args.seed === 666
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # User's environmental variables
    # !!Explain: args.gpu === 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Number of digits of precision for floating point output
    torch.set_printoptions(threshold=50000)

    # --------- File paths ---------
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    # !!Explain args.log_root === HeterSumGraph\results\log
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # !!Explain: Like this(HeterSumGraph\results\log\train_20230603_215006)
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    # !!Explain: <Logger Summarization logger (DEBUG)>
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    logger.info("[INFO] Word Embedding Dimension is %s", args.word_emb_dim)
    # !!Explain: Use 0 for padding of embeddings
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    # --------- Word Embedding from Scratch ---------
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train
    hps = args
    logger.info(hps)

    # --------- Paths of words to sentences ---------
    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")

    # --------- Using GPU OR CPU ---------
    if args.cuda and args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")
    else:
        device = torch.device("cpu")
        logger.info("[INFO] Use CPU")
    hps.device = device

    # --------- HSG(Heterogeneous Summarization Graph) Model => Single Document ---------
    if hps.model == "HSG":
        # Define Model
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        # --------- Make Dataset ---------
        """
            DATA_FILE: datasets\cnndm\\train.label.jsonl (Size = 287084)
            vocab: <module.vocabulary.Vocab object at 0x0000028B63BCFFA0
            !!Explain: the maximum sentence number of a document, each example should pad sentences to this length
            hps.doc_max_timesteps: 50
            !!Explain: the maximum token number of a sentence, each sentence should pad tokens to this length
            hps.sent_max_len: 100
            FILTER_WORD: HeterSumGraph\cache\CNNDM\\filter_word.txt
            train_w2s_path: HeterSumGraph\cache\CNNDM\\train.w2s.tfidf.jsonl
        """
        dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path)
        # --------- Train Loader ---------
        # !!Explain: collate_fn(resize to a uniform size and combine them into a single Tensor input for the neural network)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=2,
                                                   collate_fn=graph_collate_fn)
        del dataset
        # --------- Validation Loader ---------
        valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                   val_w2s_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,
                                                   collate_fn=graph_collate_fn, num_workers=32)
    # --------- HDSG(Heterogeneous Documents Summarization Graph) Model => Multiple Documents ---------
    elif hps.model == "HDSG":
        # Define Model
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
        train_w2d_path = os.path.join(args.cache_dir, "train.w2d.tfidf.jsonl")
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                  train_w2s_path, train_w2d_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,
                                                   collate_fn=graph_collate_fn)
        del dataset
        val_w2d_path = os.path.join(args.cache_dir, "val.w2d.tfidf.jsonl")
        valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                        val_w2s_path, val_w2d_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,
                                                   collate_fn=graph_collate_fn,
                                                   num_workers=32)  # Shuffle Must be False for ROUGE evaluation
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    # --------- Start Training ---------
    setup_training(model, train_loader, valid_loader, valid_dataset, hps)


if __name__ == '__main__':
    main()
