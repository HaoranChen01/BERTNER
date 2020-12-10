from transformers import BertPreTrainedModel, AdamW
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict
from torch import nn


def train(model, num_epochs, Train_data_loader, Val_data_loader,
          freezing=False, device_type='cuda', lr=3e-5, model_name='model',):
    logging.info('Commencing training!')
    torch.manual_seed(196)

    device = torch.device(device_type)
    model.to(device)

    if freezing:
        for param in model.base_model.parameters():
            param.requires_grad = False

    weight_decay = 0.01
    crf_learning_rate = 1e-3
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer_pre = list(model.bert.named_parameters())[:149]
    bert_param_optimizer_last = list(model.bert.named_parameters())[149:]
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    # lstm_param_optimizer = list(model.bi_lstm.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer_pre if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr},

        {'params': [p for n, p in bert_param_optimizer_last if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer_last if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        # {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
        #  'weight_decay': weight_decay, 'lr': crf_learning_rate},
        # {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        #  'lr': crf_learning_rate}
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # optim = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_loss, val_loss = [], []

    for epoch in range(num_epochs):
        model.train()
        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = 0
        stats['batch_size'] = 0
        progress_bar = tqdm(Train_data_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

        for i, batch in enumerate(progress_bar):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=True,
            )
            loss = outputs[0]
            loss.backward()
            optim.step()

            total_loss, batch_size = loss.item(), len(batch['labels'])
            stats['loss'] += total_loss
            stats['lr'] += optim.param_groups[0]['lr']
            stats['batch_size'] += batch_size
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        train_loss.append(stats['loss']/len(progress_bar))
        val_loss.append(validate(model, Val_data_loader, epoch, device_type=device_type))

        model.cpu()
        torch.save(model.state_dict(), "finetuned_models/{}-epoch{}.pt".format(model_name,str(epoch)))
        model.to(device)

    make_plot(train_loss, val_loss, model_name)


def validate(model, data_loader, epoch, device_type='cuda'):
    device = torch.device(device_type)
    model.to(device)
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['batch_size'] = 0
    progress_bar = tqdm(data_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
                labels=batch['labels'].to(device),
                return_dict=True,
            )
            loss = outputs[0]
            stats['valid_loss'] += loss.item()
            stats['batch_size'] += len(batch['labels'])
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

    return stats['valid_loss'] / len(progress_bar)


def predict(model, Test_data_loader, decoder, device_type='cuda'):
    device = torch.device(device_type)
    tot_tags = []
    tot_labels = []
    have_label = 'labels' in Test_data_loader.dataset[0].keys()
    model.to(device)
    model.eval()
    progress_bar = tqdm(Test_data_loader, desc='| Predicting', leave=False, disable=False)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            lb = None
            if have_label:
                lb = batch['labels'].to(device)
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
                labels=lb,
                return_dict=True,
            )
            tags = model.crf.decode(outputs.logits, batch['attention_mask'].to(device)).squeeze().cpu()
            if len(tags.shape) == 1:
                tags = tags.unsqueeze(0)
            tot_tags.append(tags)
            if have_label:
                labels = lb.cpu()
                tot_labels.append(labels)
            progress_bar.set_postfix({'number of batch': '{}'.format(i)},
                                     refresh=True)
    TAGS = torch.cat(tot_tags, 0)

    if have_label:
        LABELS = torch.cat(tot_labels, 0)
        naive_precision = compute_precision(TAGS, LABELS)
        naive_recall = compute_recall(TAGS, LABELS)
        TAGS = tensor2entity(TAGS,decoder)
        LABELS = tensor2entity(LABELS,decoder)
        precision, recall, f1 = evaluate(LABELS, TAGS)
    else:
        TAGS = tensor2entity(TAGS,decoder)
        return TAGS

    return {
        'TAGS': TAGS,
        'LABELS': LABELS,
        'naive_precision': naive_precision,
        'naive_recall': naive_recall,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def make_plot(train_loss, val_loss, model_name):
    plt.style.use('ggplot')
    plt.figure()
    plt.title('{}'.format(model_name))
    plt.xlabel('num of epoch')
    plt.ylabel('loss')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validate'])
    plt.savefig('figs//{}.png'.format(model_name))


def bert_layers_init(model, k=[9,10,11]):
    para_dict = model.state_dict()
    for key in para_dict.keys():
        if set(list(key)) & set(map(str,k)):
            nn.init.normal_(para_dict[key],mean=0,std=0.02)
    model.load_state_dict(para_dict)