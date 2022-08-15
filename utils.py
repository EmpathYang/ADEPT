from genericpath import isfile
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import csv
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'data')
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
LOG_DIR = os.path.join(CURRENT_DIR, 'log')
IMG_DIR = os.path.join(CURRENT_DIR, 'img')

SELECTED_SEAT_TEST = ['sent-weat6', 'sent-weat6b', 'sent-weat7', 'sent-weat7b', 'sent-weat8', 'sent-weat8b']
STEREOSET_SCORE = ['LM Score', 'SS Score', 'ICAT Score']

def preprocess_log_file(filename):
    file_path = os.path.join(LOG_DIR, filename)
    eval_step_loss_pair = []
    with open(file_path, 'r') as rf:
        content = rf.readlines()
        for line in content:
            line = line.strip()
            if 'global_step' in line and 'evaluate loss' in line:
                global_step, evaluate_loss = line.split(',')
                eval_step_loss_pair.append((int(global_step.split('=')[-1]), float(evaluate_loss.split('=')[-1])))
    return eval_step_loss_pair

def plot_loss_step_fig(step_loss_pair, start_end_pair, save_img_name):
    step_loss_pair = np.array(step_loss_pair)
    step, loss = step_loss_pair[start_end_pair[0]:start_end_pair[1], 0], step_loss_pair[start_end_pair[0]:start_end_pair[1], 1]

    plt.figure()
    plt.title(save_img_name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(step, loss)
    plt.savefig(os.path.join(IMG_DIR, f'{save_img_name}.png'), dpi=100)

def plot_word_correlation_gender(word_correlation_dir, layers='all', perplexity=30, result_filename='default'):
    with open(os.path.join(DATA_DIR, 'male_seat.txt'), 'r') as rf:
        male_words = rf.readlines()
        male_words = [w.strip() for w in male_words]
    with open(os.path.join(DATA_DIR, 'female_seat.txt'), 'r') as rf:
        female_words = rf.readlines()
        female_words = [w.strip() for w in female_words]
    with open(os.path.join(DATA_DIR, 'neutral_seat.txt'), 'r') as rf:
        neutral_words = rf.readlines()
        neutral_words = [w.strip() for w in neutral_words]
    word_embedding = torch.load(os.path.join(word_correlation_dir, 'word_embedding_30.bin'))
    words = word_embedding.keys()
    if layers == 'all':
        vectors = [v.view(-1).tolist() for v in word_embedding.values()]
    else:
        vectors = [v[torch.tensor(layers), :].view(-1).tolist() for v in word_embedding.values()]
    m_i = []
    f_i = []
    n_i = []
    for i, w in enumerate(words):
        if w in male_words:
            m_i.append(i)
        elif w in female_words:
            f_i.append(i)
        elif w in neutral_words:
            n_i.append(i)
        else:
            print(f'Bad word: {w}')
    m_i = np.array(m_i)
    f_i = np.array(f_i)
    n_i = np.array(n_i)
    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=46)
    result = tsne.fit_transform(vectors)
    X = result[..., 0]
    Y = result[..., 1]
    plt.figure()
    plt.scatter(X[m_i], Y[m_i], c='xkcd:sky blue')
    plt.scatter(X[f_i], Y[f_i], c='xkcd:dark pink')
    plt.scatter(X[n_i], Y[n_i], c='xkcd:beige')
    for w, x, y in zip(words, X, Y):
        plt.annotate(w, (x, y), fontsize=5)
    plt.title('t-SNE')
    plt.savefig(os.path.join(word_correlation_dir, f't-SNE[layer]{layers[0]}-{result_filename}.png'), dpi=1000)
    plt.close()

def plot_word_correlation_religion(word_correlation_dir, layers='all', perplexity=80, result_filename='default'):
    with open(os.path.join(DATA_DIR, 'jewish.txt'), 'r') as rf:
        jewish_words = rf.readlines()
        jewish_words = [w.strip() for w in jewish_words]
    with open(os.path.join(DATA_DIR, 'christian.txt'), 'r') as rf:
        christian_words = rf.readlines()
        christian_words = [w.strip() for w in christian_words]
    with open(os.path.join(DATA_DIR, 'muslim.txt'), 'r') as rf:
        muslim_words = rf.readlines()
        muslim_words = [w.strip() for w in muslim_words]
    with open(os.path.join(DATA_DIR, 'neutral_seat.txt'), 'r') as rf:
        neutral_words = rf.readlines()
        neutral_words = [w.strip() for w in neutral_words]
    word_embedding = torch.load(os.path.join(word_correlation_dir, 'word_embedding_30.bin'))
    words = word_embedding.keys()
    if layers == 'all':
        vectors = [v.view(-1).tolist() for v in word_embedding.values()]
    else:
        vectors = [v[torch.tensor(layers), :].view(-1).tolist() for v in word_embedding.values()]
    j_i = []
    c_i = []
    m_i = []
    n_i = []
    for i, w in enumerate(words):
        if w in jewish_words:
            j_i.append(i)
        elif w in christian_words:
            c_i.append(i)
        elif w in muslim_words:
            m_i.append(i)
        elif w in neutral_words:
            n_i.append(i)
        else:
            print(f'Bad word: {w}')
    j_i = np.array(j_i)
    c_i = np.array(c_i)
    m_i = np.array(m_i)
    n_i = np.array(n_i)
    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=46)
    result = tsne.fit_transform(vectors)
    X = result[..., 0]
    Y = result[..., 1]
    plt.figure()
    plt.scatter(X[j_i], Y[j_i], c='xkcd:goldenrod')
    plt.scatter(X[c_i], Y[c_i], c='xkcd:robin\'s egg blue')
    plt.scatter(X[m_i], Y[m_i], c='xkcd:olive green')
    plt.scatter(X[n_i], Y[n_i], c='xkcd:light grey')
    for w, x, y in zip(words, X, Y):
        plt.annotate(w, (x, y), fontsize=5)
    plt.title('t-SNE')
    plt.savefig(os.path.join(word_correlation_dir, f't-SNE[layer]{layers[0]}-{result_filename}.png'), dpi=1000)
    plt.close()

def rearrange_SEAT_result(filedir='/home/y/bias-bench/results/seat', result_filename='seat_all'):
    with open(os.path.join(RESULT_DIR, f'{result_filename}.csv'), 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow([''] + SELECTED_SEAT_TEST)
        for path in os.listdir(filedir):
            if os.path.isfile(os.path.join(filedir, path)):
                with open(os.path.join(filedir, path), 'r') as rf:
                    result = json.load(rf)
                scores = []
                for item in result:
                    if item['test'] in SELECTED_SEAT_TEST:
                        scores.append('{0:.3f}({1:.3f})'.format(item['effect_size'], item['p_value']))
                writer.writerow([path] + scores)

def rearrange_STEREOSET_result(filepath='/home/y/bias-bench/results/stereoset/all.json', result_filename='stereoset_all'):
    with open(os.path.join(RESULT_DIR, f'{result_filename}.csv'), 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow([''] + STEREOSET_SCORE)
        writer.writerow(['Ideal', 100, 50, 100])
        writer.writerow(['Stereotyped', '-', 100, 0])
        writer.writerow(['Random', 50, 50, 50])
        with open(filepath, 'r') as rf:
            result = json.load(rf)
        for key in result.keys():
            scores = [result[key]['intrasentence']['religion'][s] for s in STEREOSET_SCORE]
            scores = ['{0:.3f}'.format(s) for s in scores]
            writer.writerow([key+':religion'] + scores)
            scores = [result[key]['intrasentence']['overall'][s] for s in STEREOSET_SCORE]
            scores = ['{0:.3f}'.format(s) for s in scores]
            writer.writerow([key+':overall'] + scores)

def rearrange_CROWS_PAIRS_result(filedir='/home/y/bias-bench/results/crows/ab-test', result_filename='crows_pairs_all'):
    with open(os.path.join(RESULT_DIR, f'{result_filename}.csv'), 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow(['', 'score(S)'])
        writer.writerow(['Ideal', 50])
        for path in os.listdir(filedir):
            score = open(os.path.join(filedir, path), 'r').readlines()[0]
            writer.writerow([path, score])

def plot():
    # log_filename = 'prompt_tuning/2022-07-11-05-21-14' # prompt-tuning
    # save_filename = 'bert-large-uncased-prompt-tuning'
    # log_filename = 'finetuning/2022-07-09-05-35-19' # finetuning
    # save_filename = 'bert-large-uncased-finetuning_42'
    # log_filename = 'finetuning/2022-07-13-06-49-39' # finetuning
    # save_filename = 'bert-large-uncased-finetuning_42_test'
    # log_filename = 'finetuning/2022-07-11-07-53-40' # finetuning
    # save_filename = 'bert-large-uncased-finetuning_137'
    # log_filename = 'finetuning/2022-07-13-06-11-02'
    # save_filename = 'bert-large-uncased-finetuning_605' #finetuning
    # log_filename = 'finetuning_representation/30/2022-07-15-10-10-38' # finetuning_representation
    # save_filename = 'bert-large-uncased-finetuning_representation'
    # log_filename = 'finetuning/2022-07-29-05-22-55'
    # save_filename = 'DPCE'
    log_filename = 'prompt_tuning_bias/15/2022-08-08-02-49-55'
    save_filename = 'ADEPT'
    # log_filename = 'test_finetuning/2022-07-19-05-23-17' # finetuning
    # save_filename = 'bert-large-uncased-test_finetuning'
    eval_step_loss_pair = preprocess_log_file(log_filename + '.log')
    plot_loss_step_fig(eval_step_loss_pair, (0, 20), save_filename)

def sample_sentences_from_bookcorpus(seed, sample_size):
    rf1 = open(os.path.join(CORPUS_DIR, 'books_large_p1.txt'), 'r')
    rf2 = open(os.path.join(CORPUS_DIR, 'books_large_p2.txt'), 'r')
    content1 = rf1.readlines()
    content2 = rf2.readlines()
    rf1.close()
    rf2.close()
    content = content1 + content2
    if sample_size == 'all':
        with open(os.path.join(DATA_DIR, 'sampled_corpus', 'all.txt'), 'w') as wf:
            wf.writelines(content)
        return
    random.seed(seed)
    index = random.choices(range(len(content)), k=sample_size)
    sampled_corpus = []
    for i in index:
        sampled_corpus.append(content[i])
    os.makedirs(os.path.join(DATA_DIR, 'sampled_corpus'), exist_ok=True)
    with open(os.path.join(DATA_DIR, 'sampled_corpus', f'[seed]{seed}[sample_size]{sample_size}.txt'), 'w') as wf:
        wf.writelines(sampled_corpus)

def match_attribute_words(attribute_filename_list):
    attribute_list = []
    for attribute_filename in attribute_filename_list:
        with open(os.path.join(DATA_DIR, attribute_filename), 'r') as rf:
            l = [word.strip() for word in rf.readlines()]
            attribute_list.append(l)
    matched_attribute_word = {}
    for i in range(len(attribute_list)):
        for j in range(len(attribute_list)):
            if j != i:
                for k in range(len(attribute_list[0])):
                    if attribute_list[i][k] not in matched_attribute_word.keys():
                        matched_attribute_word[attribute_list[i][k]] = []
                    if attribute_list[j][k] not in matched_attribute_word[attribute_list[i][k]]:
                        matched_attribute_word[attribute_list[i][k]].append(attribute_list[j][k])
    with open(os.path.join(DATA_DIR, 'matched_attribute_word.json'), 'w') as wf:
        json.dump(matched_attribute_word, wf)
if __name__ == '__main__':
    # plot()
    # perplexities = [1, 10, 20, 30, 35, 40, 45, 50, 100, 150, 200]
    # layers = [[i] for i in range(25)]
    # layers += ['all']
    # for p in perplexities:
    #     for l in layers:
    #         plot_word_vector('/home/y/context-debias/debiased_models/42/bert-large-uncased/word_vector/word_vector_30.bin', layers=l, perplexity=p)
    # layers = [[i] for i in range(25)]
    # layers = [[24]]
    # for l in layers:
    #     plot_word_vector_religion('/home/y/context-debias/debiased_models/42/bert-large-uncased/word_vector/word_vector_30.bin', layers=l, result_filename='0b-r-test')
    #     plot_word_vector_religion('/home/y/context-debias/debiased_models/42/bert-large-uncased/word_vector/prompt/word_vector_30.bin', layers=l, result_filename='0p-r-test')
    # pass
    rearrange_STEREOSET_result()
    # rearrange_CROWS_PAIRS_result()
    # rearrange_SEAT_result()