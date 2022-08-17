import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(CURRENT_DIR, 'corpus')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
LOG_DIR = os.path.join(CURRENT_DIR, 'log')
IMG_DIR = os.path.join(CURRENT_DIR, 'img')


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
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMG_DIR, f'{save_img_name}.png'), dpi=100)

def plot_word_correlation_gender(word_correlation_dir, layers='all', perplexity=30, result_filename='default'):
    with open(os.path.join(DATA_DIR, 'male.txt'), 'r') as rf:
        male_words = rf.readlines()
        male_words = [w.strip() for w in male_words]
    with open(os.path.join(DATA_DIR, 'female.txt'), 'r') as rf:
        female_words = rf.readlines()
        female_words = [w.strip() for w in female_words]
    with open(os.path.join(DATA_DIR, 'neutral.txt'), 'r') as rf:
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
    with open(os.path.join(DATA_DIR, 'judaism.txt'), 'r') as rf:
        judaism_words = rf.readlines()
        judaism_words = [w.strip() for w in judaism_words]
    with open(os.path.join(DATA_DIR, 'christianity.txt'), 'r') as rf:
        christianity_words = rf.readlines()
        christianity_words = [w.strip() for w in christianity_words]
    with open(os.path.join(DATA_DIR, 'islam.txt'), 'r') as rf:
        islam_words = rf.readlines()
        islam_words = [w.strip() for w in islam_words]
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
    i_i = []
    n_i = []
    for i, w in enumerate(words):
        if w in judaism_words:
            j_i.append(i)
        elif w in christianity_words:
            c_i.append(i)
        elif w in islam_words:
            i_i.append(i)
        elif w in neutral_words:
            n_i.append(i)
        else:
            print(f'Bad word: {w}')
    j_i = np.array(j_i)
    c_i = np.array(c_i)
    i_i = np.array(i_i)
    n_i = np.array(n_i)
    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=46)
    result = tsne.fit_transform(vectors)
    X = result[..., 0]
    Y = result[..., 1]
    plt.figure()
    plt.scatter(X[j_i], Y[j_i], c='xkcd:goldenrod')
    plt.scatter(X[c_i], Y[c_i], c='xkcd:robin\'s egg blue')
    plt.scatter(X[i_i], Y[i_i], c='xkcd:olive green')
    plt.scatter(X[n_i], Y[n_i], c='xkcd:light grey')
    for w, x, y in zip(words, X, Y):
        plt.annotate(w, (x, y), fontsize=5)
    plt.title('t-SNE')
    plt.savefig(os.path.join(word_correlation_dir, f't-SNE[layer]{layers[0]}-{result_filename}.png'), dpi=1000)
    plt.close()

def plot():
    log_filename = '' # Relative path of the log file.
    save_filename = 'ADEPT'
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


if __name__ == '__main__':
    pass