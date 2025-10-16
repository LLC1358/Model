import argparse
import datetime
import os
import pickle
import time
import torch
import tqdm

import numpy as np

from data import *
from evaluate import *
from model import *

###################################################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MIND-small', help='MIND-large')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)   # 16
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty') 
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[4], help='the epoch which the learning rate decay')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--Glove_dim', type=int, default=300)
parser.add_argument('--category_dim', type=int, default=100)
parser.add_argument('--subcategory_dim', type=int, default=100)
parser.add_argument('--popularity_dim', type=int, default=100)
parser.add_argument('--recency_dim', type=int, default=100)
parser.add_argument('--n_popularity_level', type=int, default=8)
parser.add_argument('--n_recency_group', type=int, default=1+3)   # 0: padding
parser.add_argument('--num_head_n', type=int, default=16)
parser.add_argument('--num_head_u', type=int, default=24)
parser.add_argument('--num_head_c', type=int, default=24)
parser.add_argument('--head_dim', type=int, default=25)
parser.add_argument('--num_extracted_interests', type=int, default=8)
parser.add_argument('--M', type=int, default=12, help='the scaling of the unique_category_counts to compute the num_selected_interests')
parser.add_argument('--cold_start_users', type=int, default=3, help='the number of clicked news for a cold-start user')
parser.add_argument('--cold_start_users_weight', type=float, default=0.25, help='the weight of a_u for cold-start users')
parser.add_argument('--cold_start_users_threshold', type=int, default=0, help='the number of clicked news (threshold) for a cold-start user')
'''
parser.add_argument('--lambda_c', type=float, default=0.01)
'''
parser.add_argument('--aug', type=bool, default=True)
parser.add_argument('--save_path', default='model_save')
opt = parser.parse_args()
print(opt)

if opt.save_path is not None:
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + 'aug=' + str(opt.aug) + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

###################################################################################################################################################

def main():
    init_seed(2023)

    if opt.dataset == 'MIND-small':
        n_category = 19 + 1       # 18 + 1
        n_subcategory = 271 + 1   # 270 + 1
        max_history_news_click_num = 4802
    elif opt.dataset == 'MIND-large':
        n_category = 19 + 1       # 18 + 1
        n_subcategory = 286 + 1   # 293 + 1
        max_history_news_click_num = 69736

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    valid_data = pickle.load(open('datasets/' + opt.dataset + '/validation.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    news2category = pickle.load(open('datasets/' + opt.dataset + '/news2category.txt', 'rb'))
    news2subcategory = pickle.load(open('datasets/' + opt.dataset + '/news2subcategory.txt', 'rb'))
    word_embeddings = pickle.load(open('datasets/' + opt.dataset + '/word_embeddings.txt', 'rb'))
    news_title_text = pickle.load(open('datasets/' + opt.dataset + '/news_title_text.txt', 'rb'))
    news_abstract_text = pickle.load(open('datasets/' + opt.dataset + '/news_abstract_text.txt', 'rb'))
    news_popularity_label = pickle.load(open('datasets/' + opt.dataset + f'/popularity_level.txt', 'rb'))
    news_recency_label = pickle.load(open('datasets/' + opt.dataset + f'/recency_group.txt', 'rb'))
    news_click_num = pickle.load(open('datasets/' + opt.dataset + f'/history_click_num.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    valid_data = Data(valid_data, shuffle=False)
    test_data = Data(test_data, shuffle=False)

    model = MODEL(device, opt, word_embeddings, n_category, n_subcategory, max_history_news_click_num)
    model.initialize()
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    start = time.time()

    for epoch in range(opt.epoch):
        print('-------------------------------------------')
        print(f'epoch: {epoch}')
        
        # Training
        train(model, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, optimizer)

        # Validation
        (all, cold) = valid_test(model, valid_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num)
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = all
        cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10 = cold
        print('[All Users] AUC: %.4f\tMRR: %.4f\tNDCG@5: %.4f\tNDCG@10: %.4f' % (val_auc, val_mrr, val_ndcg5, val_ndcg10))
        print('[Cold-start Users] AUC: %.4f\tMRR: %.4f\tNDCG@5: %.4f\tNDCG@10: %.4f' % (cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10))
        scheduler.step()

    if opt.save_path is not None:
        save_model = os.path.join(save_dir, f'model.pt')
        torch.save(model.state_dict(), save_model)
        print('Model saved.')

     # Test
    print('--------------------------------------------------')
    model.load_state_dict(torch.load(os.path.join(save_dir, f'model.pt')))
    if opt.dataset == 'MIND-small':
        (all, cold) = valid_test(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num)
        test_auc, test_mrr, test_ndcg5, test_ndcg10 = all
        cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10 = cold
        print('[All Users] AUC: %.4f\tMRR: %.4f\tNDCG@5: %.4f\tNDCG@10: %.4f' % (test_auc, test_mrr, test_ndcg5, test_ndcg10))
        print('[Cold-start Users] AUC: %.4f\tMRR: %.4f\tNDCG@5: %.4f\tNDCG@10: %.4f' % (cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10))
    elif opt.dataset == 'MIND-large':
        prediction(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, opt.prediction_path)
        
    end = time.time()
    print("Run time: %f s" % (end - start))

###################################################################################################################################################

def train(model, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, optimizer):
    model.train()
    total_loss = []
    train_slices = train_data.generate_batch(opt.batch_size)

    for index in tqdm.tqdm(train_slices, desc='Training'):
        optimizer.zero_grad()
        scores, labels, _ = forward(model, index, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
        '''
        scores, labels, _, category_preds, category_labels = forward(model, index, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
        '''
        labels = torch.where(labels != 0)[1]
        
        ### News Recommendation Task ###
        loss_r = model.loss_function(scores, labels)
        '''
        ### News Category Classification Task ###
        category_preds_flat = category_preds.view(-1, category_preds.size(-1))   # [batch_size*(np_ratio+1), n_category]
        category_labels_flat = category_labels.view(-1)                          # [batch_size*(np_ratio+1)]
        mask = category_labels_flat != 0                                         # [batch_size*(np_ratio+1)]
        if mask.sum() > 0:
            loss_c = F.cross_entropy(category_preds_flat[mask], category_labels_flat[mask])
        else:
            loss_c = torch.tensor(0.0, device=device)

        ###############################################################
        
        ### Gradient Surgery ###
        # 1. Backward: News Recommendation Task (Main Task)
        loss_r.backward(retain_graph=True)
        g_main = []
        params = []
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
                if param.grad is not None:
                    g_main.append(param.grad.detach().clone().flatten())
                else:
                    g_main.append(torch.zeros_like(param.data).flatten())
        g_main = torch.cat(g_main)

        # 2. Backward: News Category Classification Task (Auxiliary Task)
        #optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        g_aux = []
        for param in params:
            if param.grad is not None:
                g_aux.append(param.grad.detach().clone().flatten())
            else:
                g_aux.append(torch.zeros_like(param.data).flatten())
        g_aux = torch.cat(g_aux)

        # 3. Gradient Surgery
        dot = torch.dot(g_main, g_aux)
        if dot < 0:
            g_aux_norm = g_aux.norm() ** 2 + 1e-8
            proj = (dot / g_aux_norm) * g_aux
            g_main_proj = g_main - 0.01 * proj
        else:
            g_main_proj = g_main

        # 4. Update
        idx = 0
        for param in params:
            numel = param.numel()
            param.grad = g_main_proj[idx:idx+numel].view_as(param).clone()
            idx += numel
        
        ###############################################################
        '''
        loss = loss_r #+ opt.lambda_c * loss_c
        loss.backward()
        optimizer.step()

        #break

    print(f"Loss_r: {loss_r.item():.6f}")
    '''
    print(f"Loss_r: {loss_r.item():.6f}, Loss_c: {loss_c.item():.6f}")
    '''

###################################################################################################################################################

def valid_test(model, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num):
    model.eval()
    slices = data.generate_batch(opt.batch_size)
    
    click_counts_all = []
    total_scores, total_labels, total_lengths = [], [], []
    '''
    total_category_preds, total_category_labels = [], []
    '''

    with torch.no_grad():
        for index in tqdm.tqdm(slices, desc='Evaluating'):
            history_padding, _, _ = data.get_padding(index) 
            history_mask_np = np.where(history_padding == 0, 0, 1)
            click_counts_batch = history_mask_np.sum(axis=1).tolist()
            click_counts_all.extend(click_counts_batch)
            
            scores, labels, lengths = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            '''
            scores, labels, lengths, category_preds, category_labels = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            '''
            total_scores.extend(scores.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            total_lengths.extend(lengths.cpu().numpy())

            '''
            total_category_preds.append(category_preds.view(-1, category_preds.size(-1)))
            total_category_labels.append(category_labels.view(-1))
            '''

            #break
        
    '''
    all_preds = torch.cat(total_category_preds, dim=0).argmax(dim=-1).view(-1)
    all_labels = torch.cat(total_category_labels, dim=0).view(-1)
    mask = all_labels != 0
    acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
    print(f"[Topic Classification Accuracy]: {acc:.4f}")
    ''' 

    ### All Users ###
    preds, labels = [], []
    for score, label, length in zip(total_scores, total_labels, total_lengths):
        score = np.asarray(score[:int(length)])
        label = np.asarray(label[:int(length)])
        rank = np.argsort(-score)   # 依分數由大到小排序
        labels.append(label[rank])
        preds.append(list(range(1, len(rank)+1)))
    auc, mrr, ndcg5, ndcg10 = evaluate(preds, labels)

    ### Cold-start Users ###
    preds_cs, labels_cs = [], []
    for score, label, length, cc in zip(total_scores, total_labels, total_lengths, click_counts_all):
        if cc <= opt.cold_start_users_threshold:
            score = np.asarray(score[:int(length)])
            label = np.asarray(label[:int(length)])
            rank = np.argsort(-score)
            labels_cs.append(label[rank])
            preds_cs.append(list(range(1, len(rank)+1)))

    if len(preds_cs) > 0:
        auc_cs, mrr_cs, ndcg5_cs, ndcg10_cs = evaluate(preds_cs, labels_cs)
    else:
        auc_cs = mrr_cs = ndcg5_cs = ndcg10_cs = float('nan')

    all = (auc*100, mrr*100, ndcg5*100, ndcg10*100)
    cold = (auc_cs*100, mrr_cs*100, ndcg5_cs*100, ndcg10_cs*100)
   
    return all, cold

###################################################################################################################################################

def prediction(model, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, output_path):
    model.eval()
    slices = data.generate_batch(opt.batch_size)

    with open(output_path, 'w', encoding='utf-8') as f:
        for index in tqdm.tqdm(slices, desc='Predicting'):
            scores, _, lengths = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            '''
            scores, labels, lengths, category_preds, category_labels = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            '''
            scores = scores.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()

            batch_size = len(index)
            for i in range(batch_size):
                impression_id = index[i] + 1
                length = int(lengths[i])
                sorted_indices = np.argsort(-scores[i][:length])
                
                ranking = [0] * length
                for rank, idx_ in enumerate(sorted_indices):
                    ranking[idx_] = rank + 1
                
                f.write(f"{impression_id} [{','.join(map(str, ranking))}]\n")    

            #break  

###################################################################################################################################################

def forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device):
    history_padding, candidate_padding, label_padding = data.get_padding(index)
    #neighbour_padding = semantic_similar_news[candidate_padding]
    history_mask = np.where(history_padding == 0, np.zeros_like(history_padding), np.ones_like(history_padding))

    history_category = news2category[history_padding]
    candidate_category = news2category[candidate_padding]
    #neighbour_category = news2category[neighbour_padding]

    history_subcategory = news2subcategory[history_padding]
    candidate_subcategory = news2subcategory[candidate_padding]
    #neighbour_subcategory = news2subcategory[neighbour_padding]

    history_title_word = news_title_text[history_padding]
    history_title_word_mask = np.where(history_title_word == 0, np.zeros_like(history_title_word), np.ones_like(history_title_word))
    candidate_title_word = news_title_text[candidate_padding]
    candidate_title_word_mask = np.where(candidate_title_word == 0, np.zeros_like(candidate_title_word), np.ones_like(candidate_title_word))
    #neighbour_title_word = news_title_text[neighbour_padding]
    #neighbour_title_word_mask = np.where(neighbour_title_word == 0, np.zeros_like(neighbour_title_word), np.ones_like(neighbour_title_word))

    history_abstract_word = news_abstract_text[history_padding]
    history_abstract_word_mask = np.where(history_abstract_word == 0, np.zeros_like(history_abstract_word), np.ones_like(history_abstract_word))
    candidate_abstract_word = news_abstract_text[candidate_padding]
    candidate_abstract_word_mask = np.where(candidate_abstract_word == 0, np.zeros_like(candidate_abstract_word), np.ones_like(candidate_abstract_word))
    #neighbour_abstract_word = news_abstract_text[neighbour_padding]
    #neighbour_abstract_word_mask = np.where(neighbour_abstract_word == 0, np.zeros_like(neighbour_abstract_word), np.ones_like(neighbour_abstract_word))

    candidate_popularity_label = news_popularity_label[candidate_padding]
    candidate_recency_label = news_recency_label[candidate_padding]
    history_click_num = news_click_num[history_padding]

    ###############################################################

    candidate_padding = torch.LongTensor(candidate_padding).to(device)
    label_padding = torch.LongTensor(label_padding).to(device)
    history_mask = torch.FloatTensor(history_mask).to(device)

    history_category = torch.LongTensor(history_category).to(device)
    candidate_category = torch.LongTensor(candidate_category).to(device)
    #neighbour_category = torch.LongTensor(neighbour_category).to(device)

    history_subcategory = torch.LongTensor(history_subcategory).to(device)
    candidate_subcategory = torch.LongTensor(candidate_subcategory).to(device)
    #neighbour_subcategory = torch.LongTensor(neighbour_subcategory).to(device)

    history_title_word = torch.LongTensor(history_title_word).to(device)
    history_title_word_mask = torch.LongTensor(history_title_word_mask).to(device)
    candidate_title_word = torch.LongTensor(candidate_title_word).to(device)
    candidate_title_word_mask = torch.LongTensor(candidate_title_word_mask).to(device)
    #neighbour_title_word = torch.LongTensor(neighbour_title_word).to(device)
    #neighbour_title_word_mask = torch.LongTensor(neighbour_title_word_mask).to(device)

    history_abstract_word = torch.LongTensor(history_abstract_word).to(device)
    history_abstract_word_mask = torch.LongTensor(history_abstract_word_mask).to(device)
    candidate_abstract_word = torch.LongTensor(candidate_abstract_word).to(device)
    candidate_abstract_word_mask = torch.LongTensor(candidate_abstract_word_mask).to(device)
    #neighbour_abstract_word = torch.LongTensor(neighbour_abstract_word).to(device)
    #neighbour_abstract_word_mask = torch.LongTensor(neighbour_abstract_word_mask).to(device)

    candidate_popularity_label = torch.LongTensor(candidate_popularity_label).to(device)
    candidate_recency_label = torch.LongTensor(candidate_recency_label).to(device)
    history_click_num = torch.LongTensor(history_click_num).to(device)

    ###############################################################
    
    scores, candidate_lengths = model(history_mask, candidate_padding,
                                      history_title_word, history_title_word_mask,
                                      candidate_title_word, candidate_title_word_mask,
                                      #neighbour_title_word, neighbour_title_word_mask,
                                      history_abstract_word, history_abstract_word_mask,
                                      candidate_abstract_word, candidate_abstract_word_mask,
                                      #neighbour_abstract_word, neighbour_abstract_word_mask,
                                      history_category, candidate_category, #neighbour_category,
                                      history_subcategory, candidate_subcategory, #neighbour_subcategory,
                                      candidate_popularity_label, candidate_recency_label, history_click_num)

    return scores, label_padding, candidate_lengths
    '''
    scores, candidate_lengths, category_preds = model(history_mask, candidate_padding,
                                                      history_title_word, history_title_word_mask,
                                                      candidate_title_word, candidate_title_word_mask,
                                                      history_abstract_word, history_abstract_word_mask,
                                                      candidate_abstract_word, candidate_abstract_word_mask,
                                                      history_category, candidate_category,
                                                      history_subcategory, candidate_subcategory,
                                                      candidate_popularity_label, candidate_recency_label, history_click_num)

    return scores, label_padding, candidate_lengths, category_preds, candidate_category
    '''

###################################################################################################################################################

if __name__ == '__main__':
    main()