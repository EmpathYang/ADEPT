import os
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    plt.savefig(os.path.join(IMG_DIR, f'{save_img_name}.png'), dpi=100)

log_filename = '2022-07-06-02-51-43' # prompt-tuning
save_filename = 'bert-large-uncased-prompt-tuning'
# log_filename = 'bert-large-uncased-finetuning' # fint-tuning
# save_filename = 'bert-large-uncased-finetuning'
eval_step_loss_pair = preprocess_log_file(log_filename + '.log')
plot_loss_step_fig(eval_step_loss_pair, (20, -1), save_filename)