import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from src.adversarial.uvmifgsm import VMIFGSM
from src.adversarial.jpeg_ifgm import JIFGSM
from src.adversarial.cvfgsm import CVFGSM
from src.adversarial.perc_cw import PerC_CW
from src.adversarial.asr_metric import AUC, ConditionalAverageRate
from datetime import date



class AUCTest:
    
    def __init__(self,
                attack_type,
                model,
                dataset_type,
                test_robustness,
                spatial_attack_params,
                c=1,
                lr=0.001):
        print('\nINFO: AUCTestEnvironment is initialized so c and attack_lr arguments will be overwritten\n')
        self.model = model
        self.attack_type = attack_type
        self.attack_family = self.get_attack_family(attack_type)
        if self.attack_family == 'adv_optim':
            self.c = c
            self.lr = lr
            self.n_starts = 5
        self.eps_list = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        self.dataset_type = dataset_type
        self.test_robustness = test_robustness
        self.asp_list = []
        self.eps_for_div = 0.0000001
        
        if test_robustness:
            # those determine, how much quality will be available after compression
            # i.e. 100 -> no quality loss
            # values below therefore test mild compression (as found in most apps today)
            self.compression_range = [[70], [80], [90]]
        
        if attack_type == 'vmifgsm':
            self.attack_cls = VMIFGSM
        elif attack_type == 'cvfgsm':
            self.attack_cls = CVFGSM
        elif attack_type == 'jifgsm':
            self.attack_cls = JIFGSM
        elif attack_type == 'varrcw':
            self.attack_cls = VarRCW
        elif attack_type == 'perccw':
            self.attack_cls = PerC_CW
        else:
            raise ValueError('cw_type not recognized!')
        
        report_base = './saves/reports/auc_reports'
        if not os.path.exists(report_base):
            os.mkdir(report_base)
        
        self.report_dir = report_base
        self.report_dir += '/'
        self.report_dir += date.today().isoformat() + '_'
        self.report_dir += self.attack_type
        self.report_dir += '_' + f'compression:{test_robustness}'
        self.report_dir = self.resolve_name_collision(self.report_dir)
        os.mkdir(self.report_dir)
        if self.test_robustness:
            for compression_value in self.compression_range:
                os.mkdir(f'{self.report_dir}/compr-{compression_value}')
                main_losses_csv = f'{self.report_dir}/compr-{compression_value}/run_results.csv'
                with open(main_losses_csv, 'a') as main_file:
                    main_obj = csv.writer(main_file)
                    main_obj.writerow(['eps','asr','cad per eps','asp'])
                os.mkdir(f'{self.report_dir}/compr-{compression_value}/reports')
            self.main_losses_csv = f'{self.report_dir}/compr-{self.compression_range[0]}/run_results.csv'
        else:
            os.mkdir(f'{self.report_dir}/reports/')
            self.main_losses_csv = f'{self.report_dir}/run_results.csv'
            with open(self.main_losses_csv, 'a') as main_file:
                main_obj = csv.writer(main_file)
                main_obj.writerow(['eps', 'asr', 'cad', 'asp'])
        
        with open(f'{self.report_dir}/run_params.txt', 'w') as f:
            f.write('CHOSEN PARAMS FOR RUN\n\n')
            f.write(f'attack_type : {attack_type}\n')
            f.write(f'eps_list : {",".join([str(i) for i in self.eps_list])}\n')
            for k, v in spatial_attack_params.__dict__.items():
                f.write(f'{k} : {v}')
        self.auc_metric = AUC(f'{self.report_dir}/auc_result.txt')
        
        print('Running AUC test with the following parameters:\n\n')
        print(f'\t eps_list : {",".join([str(i) for i in self.eps_list])}\n')
        print(f'\t attack_type : {attack_type}\n')
    
    def get_attack_family(self, attack_type):
        # check if attack is from grad-sign projection family or adv. optim family
        if attack_type.endswith('cw'):
            # this is ok because only CW types will be tested for adv optim
            attack_family = 'adv_optim'
        else:
            attack_family =  'grad_projection'
        return attack_family
    
    def reset(self):
        self.asp_list = []
    
    def get_compr_dir(self, compression_val):
        path_compr_dir = f'{self.report_dir}/compr-{compression_val}'
        self.main_losses_csv = path_compr_dir + '/' + 'run_results.csv'
        return path_compr_dir
    
    def set_dataset_and_model_trms(self, data_obj):
        self.data = data_obj

    def resolve_name_collision(self, path):
        enum = 0
        ori_path = path
        while os.path.exists(path):
            enum += 1
            path = ori_path + '_' + str(enum)
        return path

    def write_to_protocol_dir(self, eps, run_csv, compression, logger_run_file):
        asr_run, cad_run, asp_run = self.get_asr_from_run(logger_run_file)
        with open(self.main_losses_csv, 'a') as main_file:
            main_obj = csv.writer(main_file)
            if self.attack_family == 'adv_optim':
                results = self.get_results_from_run_over_eps_list(logger_run_file, asr_run)
                for row in results:
                    main_obj.writerow(row)
            else:
                main_obj.writerow([eps, asr_run, cad_run, asp_run])
        return asp_run

    def get_results_from_run_over_eps_list(self, run_dir, asr_run):
        results = []
        run_path = run_dir.split('/')[:-1]
        split_run_dir = run_dir.split('/')[-1].split('_')[1:3]
        run_base = '_'.join(split_run_dir)
        base_name = run_base + '_' + 'base'
        base_path = '/'.join(run_path) + '/' + base_name
        for eps in self.eps_list:
            cad_by_eps = ConditionalAverageRate(path=run_dir, basepath=base_path, eps=eps)
            cad = cad_by_eps()
            asp = asr_run / (cad + self.eps_for_div)
            results.append([eps, asr_run, cad, asp])
        return results

    def get_asr_from_run(self, run_dir):
        
        with open(run_dir + '/' + 'results.txt', 'r') as results_file:
            for line in results_file:
                if line.startswith('ASR'):
                    _, asr = line.strip().split(':')
                elif line.startswith('ConditionalAverageRate'):
                    _, cad = line.strip().split(':')
                elif line.startswith('ASP'):
                    _, asp = line.strip().split(':')

        return float(asr), float(cad), float(asp)
    
    def plot_roc(self, is_adv_optim_run=False, compression_val=False):

        if is_adv_optim_run:
            """if compression_val:
                path = self.main_losses_csv + f'compr-{compression_val}'
            else:
                path = self.main_losses_csv
            self.asp_list = []"""
            with open(self.main_losses_csv, 'r') as f:
                run_results = csv.reader(f)
                next(run_results)
                for line in run_results:
                    self.asp_list.append(float(line[-1]))
            
        plt.plot(self.eps_list, self.asp_list, label=f'compr:{compression_val}')
        plt.legend(loc="upper right")
    
        if compression_val:
            plt.title(f'ASP curve for type:  {self.attack_type}. Compression:{compression_val}', weight='bold')
            plt.savefig(f'{self.report_dir}/compr-{compression_val}_plots.png')
        else:
            plt.title(f'ASP curve for type:  {self.attack_type}', weight='bold')    
            plt.savefig(f'{self.report_dir}/plots.png')
            
        #plt.show()


class AUCComparison:
    
    """Class that gets two dirs of AUC experiments and maps 
    the ROC of both to one graph along with their AUC 
    """
    
    def __init__(self, dir_attack_a, dir_attack_b, comparison_type, comparison_basis):
        self.dir_attack_a = dir_attack_a
        self.dir_attack_b = dir_attack_b
        self.comparison_type = comparison_type
        self.comparison_basis = comparison_basis
        self.path_base = './saves/reports/auc_reports/'
        self.auc_metric = AUC()
        self.attack_names = {}
        
        self.check_if_both_used_compression()
    
    def check_if_both_used_compression(self):
        
        split_a = self.dir_attack_a.split('_')
        split_b = self.dir_attack_b.split('_')
        self.attack_names['a'] = split_a[1]
        self.attack_names['b'] = split_b[1]
        
        for str_a, str_b in zip(split_a, split_b):
            if str_a.startswith('compression'):
                used_comp_a = str_a.split(':')
                used_comp_b = str_b.split(':')
                if used_comp_a != used_comp_b:
                    raise ValueError('either both dirs will have to have compression or none of them.')
    
    def process_dirs(self, save_fig=False):
        
        list_ax, list_ay, list_bx, list_by = self.get_data()
        auc_a = self.auc_metric(list_ax, list_ay)
        auc_b = self.auc_metric(list_bx, list_by)
        
        plt.plot(list_ax, list_ay, color='red', label=f'{self.attack_names["a"]} : {round(auc_a, 3)}')
        plt.plot(list_bx, list_by, color='blue', label=f'{self.attack_names["b"]} : {round(auc_b, 3)}')
        plt.legend(loc='upper right')
        if save_fig:
            plt.savefig(f'./saves/reports/auc_reports/{self.attack_names["a"]}-{self.attack_names["b"]}:{self.comparison_type}')
        plt.show()
            
    def get_data(self):
        
        if self.comparison_type == 'compression_rate':
            list_ax, list_ay = self.get_compression_data(self.dir_attack_a)
            list_bx, list_by = self.get_compression_data(self.dir_attack_b)
        elif self.comparison_type == 'eps':
            list_ax, list_ay = self.get_eps_data(self.dir_attack_a)
            list_bx, list_by = self.get_eps_data(self.dir_attack_b)
        
        return list_ax, list_ay, list_bx, list_by
            
    def get_compression_data(self, dir):
        # compression - asp
        compression_to_asp = {}
        
        for entry in os.listdir(self.path_base + dir):
            if os.path.isdir(self.path_base + dir + '/' + entry):
                compression_rate = entry.split('-')[1]
                with open(self.path_base + dir + '/' + entry + '/' + 'run_results.csv', 'r') as results_csv:
                    results_obj = csv.reader(results_csv)
                    next(results_obj)
                    for line in results_obj:
                        if float(line[0]) == self.comparison_basis:
                            asp_for_compression = float(line[-1])
                            compression_to_asp[compression_rate] = asp_for_compression
        
        sorted_items = sorted([(int(x),y)for x,y in compression_to_asp.items()], key=lambda x: x[0])
        compression_rates_list = [i[0]/100 for i in sorted_items]
        asp_list = [i[1] for i in sorted_items]
        return compression_rates_list, asp_list
    
    def get_eps_data(self, dir):
        # eps - asp
        eps_list = []
        asp_list = []
        is_compression_exp = self.check_params_for_compression(dir)
        if is_compression_exp:
            with open(self.path_base + dir + '/' + f'compr-{self.comparison_basis}' + '/' + 'run_results.csv', 'r') as results_csv:
                results_obj = csv.reader(results_csv)
                next(results_obj)
                for line in results_obj:
                    eps_list.append(float(line[0]))
                    asp_list.append(float(line[-1]))
        else:
            with open(self.path_base + dir + '/' + 'run_results.csv', 'r') as results_csv:
                results_obj = csv.reader(results_csv)
                next(results_obj)
                for line in results_obj:
                    eps_list.append(float(line[0]))
                    asp_list.append(float(line[-1]))
        return eps_list, asp_list
            
    def check_params_for_compression(self, dir):
        is_compression_exp = False
        with open(self.path_base + dir + '/' + 'run_params.txt') as f:
            for line in f:
                if line.startswith('compression'):
                    value = line.strip().split(':')[-1]
                    if value == 'True':
                        is_compression_exp = True
                    else:
                        is_compression_exp = False         
        return is_compression_exp