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

_DEBUG_FLAG_ = False


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, hps):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """

    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def run_training(model, train_loader, valid_loader, valset, hps, train_dir):
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

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            iter_start_time = time.time()

            model.train()

            G = G.to(hps.device)
            outputs = model.forward(G)  # [n_snodes, 2]
            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
            G.nodes[snode_id].data["loss"] = criterion(outputs, label.to(hps.device)).unsqueeze(-1)  # [n_nodes, 1]
            loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
            loss = loss.mean()

            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        # logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            optimizer.zero_grad()
            loss.backward()
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            optimizer.step()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                            .format(i, (time.time() - iter_start_time), float(train_loss / 100)))
                train_loss = 0.0

        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F,
                                                              non_descent_cnt, saveNo)

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            return


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo):
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
    eval_dir = os.path.join(hps.save_root, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model.eval()

    iter_start_time = time.time()

    with torch.no_grad():
        tester = SLTester(model, hps.m)
        for i, (G, index) in enumerate(loader):
            G = G.to(hps.device)

            tester.evaluation(G, index, valset)

    running_avg_loss = tester.running_avg_loss

    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                      float(running_avg_loss)))
    log_score(scores_all=scores_all)
    tester.getMetric()
    F = tester.labelMetric

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (
                saveNo % 3))  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < F:
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel')  # this is where checkpoints of best models are saved
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F),
                        float(best_F), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s', float(F),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo


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

    # --------- HSG(Heterogeneous Service Graph) Model ---------
    if hps.model == "HSG":
        # Define Model
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
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
        print('ٍErfaaan')
        raise Exception("Sorry, no numbers below zero")
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=2,
                                                   collate_fn=graph_collate_fn)
        del dataset
        valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                   val_w2s_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,
                                                   collate_fn=graph_collate_fn, num_workers=32)
    # --------- HDSG(Hetero-dielectric single gate) Model ---------
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

    setup_training(model, train_loader, valid_loader, valid_dataset, hps)


if __name__ == '__main__':
    main()
