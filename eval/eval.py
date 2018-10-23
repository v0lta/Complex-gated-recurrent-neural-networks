import pickle
import numpy as np
import scipy.stats as scistats
from helper_module import *
import matplotlib.pyplot as plt
np.set_printoptions(6, suppress=False, formatter=None)
# np.set_printoptions(formatter={'float': lambda x: format(x, '6.3e')})


def print_analysis(logs, threshold, plot=True):
        regex = r"_gate_activation_(.+?)_nat_grad"

        double_gate = False
        urnn = False
        convergence_lst = []
        for log in logs:
            matches = re.finditer(regex, log[1], re.MULTILINE)
            gat_act_str = None
            for matchNum, match in enumerate(matches):
                gat_act_str = match.groups()[0]
                break
            if gat_act_str is None:
                    urnn = True
            if plot:
                plt.semilogy(log[0][0], log[0][1])

            for pos, y_value in enumerate(log[0][1]):
                if y_value < threshold:
                    if gat_act_str is not None:
                        convergence_lst.append([gat_act_str, pos])
                    else:
                        convergence_lst.append(['uRNN', pos])
                    break

            if gat_act_str == 'real_mod_sigmoid_beta':
                double_gate = True
            # debug_here()
        if plot:
            plt.show()

        # organise the convergence_list according to gate activations.
        gate_phase_hirose_lst = []
        mod_sigmoid_prod_lst = []
        mod_sigmoid_sum_lst = []
        mod_sigmoid_lst = []
        mod_sigmoid_beta_lst = []
        mod_sigmoid_gamma_lst = []
        real_mod_sigmoid_beta_lst = []
        urnn_lst = []
        for value in convergence_lst:
            if value[0] == 'gate_phase_hirose':
                gate_phase_hirose_lst.append(value[1])
            if value[0] == 'mod_sigmoid_prod':
                mod_sigmoid_prod_lst.append(value[1])
            if value[0] == 'mod_sigmoid_sum':
                mod_sigmoid_sum_lst.append(value[1])
            if value[0] == 'mod_sigmoid':
                mod_sigmoid_lst.append(value[1])
            if value[0] == 'mod_sigmoid_beta':
                mod_sigmoid_beta_lst.append(value[1])
            if value[0] == 'mod_sigmoid_gamma':
                mod_sigmoid_gamma_lst.append(value[1])
            if value[0] == 'real_mod_sigmoid_beta':
                real_mod_sigmoid_beta_lst.append(value[1])
            if value[0] == 'uRNN':
                urnn_lst.append(value[1])

        # gate_phase_hirose_lst.append(18000)
        results = [gate_phase_hirose_lst, mod_sigmoid_prod_lst,
                   mod_sigmoid_sum_lst, mod_sigmoid_lst,
                   mod_sigmoid_beta_lst, mod_sigmoid_gamma_lst]

        if double_gate:
            print('exp              ', 'iterations         ', '% converged')
            conv_rate = len(real_mod_sigmoid_beta_lst)/20.0
            if len(real_mod_sigmoid_beta_lst) > 0:
                print('real_mod_sigmoid_beta', np.mean(real_mod_sigmoid_beta_lst),
                      '        ', conv_rate)
            else:
                print('real_mod_sigmoid_beta', 'nan', '        ', conv_rate)
            return real_mod_sigmoid_beta_lst
        elif urnn:
            # debug_here()
            conv_rate_g = len(mod_sigmoid_beta_lst)/20.0
            conv_rate_urnn = len(urnn_lst)/20.0
            print('exp              ', 'iterations         ', '% converged')
            if len(mod_sigmoid_beta_lst) > 0:
                print('mod_sigmoid_beta', np.mean(mod_sigmoid_beta_lst),
                      conv_rate_g)
            else:
                print('mod_sigmoid_beta', 'nan', conv_rate_g)
            if len(urnn_lst) > 0:
                print('urnn                 ', np.mean(urnn_lst), conv_rate_urnn)
            else:
                print('urnn                 ', 'nan', conv_rate_urnn)
        else:
            print('exp              ', 'iterations         ', '% converged')
            for no, exp in enumerate(exp_names):
                conv_rate = len(results[no])/20.0
                if len(results[no]) > 0:
                    print(exp, np.mean(results[no]), '        ', conv_rate)
                else:
                    print(exp, 'nan', '        ', conv_rate)

            return results


exp_names = ['gate_phase_hirose',
             'mod_sigmoid_prod ',
             'mod_sigmoid_sum  ',
             'mod_sigmoid      ',
             'mod_sigmoid_beta ',
             'mod_sigmoid_gamma']

if 1:

    print('baseline adding problem', 0.167)
    print('baseline memory problem', 0.077)

    print('~~~~~~~~~~~~~~~~ Gate variation experiments ~~~~~~~~~~~~~~~~')
    # sum gate variation study.
    adding_path = '../logs/gate_variation_study_test/' + 'exp_res.pkl'
    adding_exps = pickle.load(open(adding_path, 'rb'))
    adding_data = np.array(adding_exps[0])

    mean_train = np.mean(adding_data, axis=1)
    median_train = np.median(adding_data, axis=1)
    mode_train = scistats.mode(adding_data, axis=1)[0]
    var_train = np.var(adding_data, axis=1)
    print('adding             mean,                median,                mode,\
                   sigma')
    for exp_no, exp_name in enumerate(exp_names):
        print(exp_name, mean_train[exp_no, :], median_train[exp_no, :],
              mode_train[exp_no, :], var_train[exp_no, :])

    # memory gate variation study.
    memory_path = '../logs/gate_variation_study_test_2/' + 'exp_res.pkl'
    memory_exps = pickle.load(open(memory_path, 'rb'))
    memory_data = np.array(memory_exps[0])

    mean_train = np.mean(memory_data, axis=1)
    median_train = np.median(memory_data, axis=1)
    mode_train = scistats.mode(memory_data, axis=1)[0]
    var_train = np.var(memory_data, axis=1)
    print('memory             mean,                median,                mode,\
                   sigma')
    for exp_no, exp_name in enumerate(exp_names):
        print(exp_name, mean_train[exp_no, :], median_train[exp_no, :],
              mode_train[exp_no, :], var_train[exp_no, :])
    print(' ')

# log value based analysis. Gate variation study
if 1:
    # adding
    print('------------------- adding problem logs --------------------')
    adding_path = '../logs/gate_variation_study_test/'
    logs = return_logs(path=adding_path, window_size=50, vtag='mse')
    adding_res = print_analysis(logs, threshold=0.01)

    # memory
    print('------------------- memory problem logs --------------------')
    memory_path = '../logs/gate_variation_study_test_2/'
    logs = return_logs(path=memory_path, window_size=50, vtag='cross_entropy')
    mem_res = print_analysis(logs, threshold=5e-7)


if 1:
    print('~~~~~~~~~~~~~~~~~~~~~ Gate necessity ~~~~~~~~~~~~~~~~~~~~~~~')
    adding_path = '../logs/gate_study_t250_2/' + 'test_res.pkl'
    adding_exps = pickle.load(open(adding_path, 'rb'))
    adding_data_gated = np.array(adding_exps[0])
    adding_data_ungated = np.array(adding_exps[1])
    print('adding             mean,                median,                mode,\
                   sigma')
    print('mod_sigmoid_beta ', np.mean(adding_data_gated, axis=0),
          np.median(adding_data_gated, axis=0),
          scistats.mode(adding_data_gated, axis=0)[0],
          np.var(adding_data_gated, axis=0))
    print('unitary_evolution', np.mean(adding_data_ungated, axis=0),
          np.median(adding_data_ungated, axis=0),
          scistats.mode(adding_data_ungated, axis=0)[0],
          np.var(adding_data_ungated, axis=0),)

    memory_path = '../logs/gate_study_t250_3/' + 'test_res.pkl'
    memory_exps = pickle.load(open(memory_path, 'rb'))
    memory_data_gated = np.array(memory_exps[0])
    memory_data_ungated = np.array(memory_exps[1])
    print('memory             mean,                median,                mode,\
                   sigma')
    print('mod_sigmoid_beta ', np.mean(memory_data_gated, axis=0),
          np.median(memory_data_gated, axis=0),
          scistats.mode(memory_data_gated, axis=0)[0],
          np.var(memory_data_gated, axis=0))
    print('unitary_evolution', np.mean(memory_data_ungated, axis=0),
          np.median(memory_data_ungated, axis=0),
          scistats.mode(memory_data_ungated, axis=0)[0],
          np.var(memory_data_ungated, axis=0))

# log value based analysis. Gate necessity study
if 1:
    # adding
    print('------------------- adding problem logs --------------------')
    adding_path = '../logs/gate_study_t250_2/'
    logs = return_logs(path=adding_path, window_size=50, vtag='mse')
    adding_res = print_analysis(logs, threshold=0.01)

    # memory
    print('------------------- memory problem logs -------------------')
    memory_path = '../logs/gate_study_t250_3/'
    logs = return_logs(path=memory_path, window_size=50, vtag='cross_entropy')
    mem_res = print_analysis(logs, threshold=5e-7)

if 1:
    print('~~~~~~~~~~~~~~~~~~~~~~~ Real double ~~~~~~~~~~~~~~~~~~~~~~~~')

    adding_path = '../logs/add_gate_study_t250_mod_sig_beta_real/' \
                  + 'test_res.pkl'
    adding_exps = pickle.load(open(adding_path, 'rb'))
    adding_data = np.array(adding_exps[0])

    mean_train = np.mean(adding_data, axis=0)
    print('adding')
    print('real_beta', mean_train)

    # memory gate variation study.
    memory_path = '../logs/mem_gate_study_t250_mod_sig_beta_real/' \
                  + 'test_res.pkl'
    memory_exps = pickle.load(open(memory_path, 'rb'))
    memory_data = np.array(memory_exps[0])

    mean_train = np.mean(memory_data, axis=0)
    print('memory')
    print('real_beta', mean_train)
    print(' ')

# log value based analysis. Gate variation study
if 1:
    # adding
    print('------------------- adding problem logs --------------------')
    adding_path = '../logs/add_gate_study_t250_mod_sig_beta_real/'
    logs = return_logs(path=adding_path, window_size=50, vtag='mse')
    adding_res = print_analysis(logs, threshold=0.01)

    # memory
    print('------------------- memory problem logs -------------------')
    memory_path = '../logs/mem_gate_study_t250_mod_sig_beta_real/'
    logs = return_logs(path=memory_path, window_size=50, vtag='cross_entropy')
    mem_res = print_analysis(logs, threshold=5e-7)
