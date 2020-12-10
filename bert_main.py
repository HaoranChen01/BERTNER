from transformers import BertTokenizer, BertConfig, AdamW
from utils import *
from datasets import BertNerDataset, HundsunDataset
from torch.utils.data import DataLoader, random_split
from models import BertBilstmCRF, BertTcnCRF
import os
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from train import *
from datetime import datetime
import argparse
import logging


def rank_train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    ############################################################

    torch.manual_seed(0)

    config_path = 'chinese-bert-wwm-ext/config.json'
    bert_config = BertConfig.from_json_file(config_path)
    tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext")
    bert_config.num_labels = 7
    model = BertTcnCRF.from_pretrained('chinese-bert-wwm-ext', config=bert_config)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 32

    optimizer = AdamW(model.parameters(), lr=5e-5)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.train()
    ###############################################################

    # Data loading code
    train_dataset = torch.load('data//liu_data//TrainDataset.pt')
    val_dataset = torch.load('data//liu_data//ValDataset.pt')

    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        ##############################
        shuffle=False,  #
        ##############################
        num_workers=0,
        pin_memory=True,
        #############################
        sampler=train_sampler)  #
    #############################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            token_type_ids = batch['token_type_ids'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=True,
            )
            loss = outputs[0]

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        model.cpu()
        torch.save(model.state_dict(), "finetuned_models/{}.pt".format('model-ddp'))


def DDP_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.27.28.196'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(rank_train, nprocs=args.gpus, args=(args,))
    #########################################################


if __name__ == '__main__':
    # DDP_train()

    config_path = 'chinese-bert-wwm-ext/config.json'
    bert_config = BertConfig.from_json_file(config_path)
    tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext")
    bert_config.num_labels = 21
    logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


    # ner_model = BertBilstmCRF.from_pretrained('chinese-bert-wwm-ext', config=bert_config)
    # bert_layers_init(model=ner_model,k=[9,10,11])
    # TrainDataset = torch.load('cluener_public/train.pt')
    # ValDataset = torch.load('cluener_public/dev.pt')
    # Train_data_loader = DataLoader(TrainDataset, batch_size=64, shuffle=False)
    # Val_data_loader = DataLoader(ValDataset, batch_size=128, shuffle=False)
    # train(
    #     model=ner_model,
    #     num_epochs=10,
    #     Train_data_loader=Train_data_loader,
    #     Val_data_loader=Val_data_loader,
    #     freezing=False,
    #     device_type='cuda:3',
    #     lr=3e-5,
    #     model_name='model',
    # )


    ner_model = BertBilstmCRF(config=bert_config)
    TestDataset = torch.load('cluener_public/dev.pt')
    Test_data_loader = DataLoader(TestDataset, batch_size=128, shuffle=False)
    model_dict = torch.load('finetuned_models/model-epoch4.pt')
    # ddp_model_dict = torch.load('finetuned_models/model-ddp.pt')
    # model_dict = OrderedDict()
    # for k, v in ddp_model_dict.items():
    #     model_dict[k[7:]] = v
    ner_model.load_state_dict(model_dict)
    decoder = DECODER().get('glue_t2e')
    predict_result = predict(ner_model, Test_data_loader, decoder, device_type='cuda:3')


    # ner_model = BertTcnCRF(config=bert_config)
    # contents = read_xlsx(xlsx_dir='data/Train_Data.xlsx',col=[2])[:100]
    # encodings = tokenizer(contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    # TestDataset = HundsunDataset(encodings)
    # Test_data_loader = DataLoader(TestDataset, batch_size=32, shuffle=False)
    # ner_model.load_state_dict(torch.load('finetuned_models/model.pt'))
    # decoder = DECODER().get('glue_t2e')
    # predict_result = predict(ner_model, Test_data_loader, decoder, device_type='cuda:3')