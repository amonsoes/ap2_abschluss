import argparse
import csv
import matplotlib.pyplot as plt

class ResultsDict:
    
    def __init__(self, dir, param, constant, compr):
        self.param = param
        self.constant = constant
        self.compr = compr
        self.results_dict = self.get_results_dict(dir)

    def get_results_dict(self, dir):

        results_dict = {
            'param' : [],
            'cad' : [],
            'asr' : [],   
        }
        
        res_csv_base = dir + '/' + '-'.join(['compr', self.compr]) + '/' + 'run_losses.csv'
        with open(res_csv_base, 'r') as run_losses_f:
            run_losses_obj = csv.reader(run_losses_f)
            next(run_losses_obj)
            for line in run_losses_obj:
                c,attack_lr,_,_,_,asr,cad = line
                c = float(c)
                attack_lr = float(attack_lr)
                asr = float(asr)
                cad = float(cad)
                if self.param == 'c':
                    if attack_lr == self.constant:
                        results_dict['param'].append(c)
                        results_dict['asr'].append(asr)
                        results_dict['cad'].append(cad)
                elif self.param == 'lr':
                    if c == self.constant:
                        results_dict['param'].append(attack_lr)
                        results_dict['asr'].append(asr)
                        results_dict['cad'].append(cad)
        return results_dict
    
    def generate_plots(self):
        plt_asr = plt.plot(self.results_dict['param'], self.results_dict['asr'], color='blue', label='ASR')
        plt_cad = plt.plot(self.results_dict['param'], self.results_dict['cad'], color='red', linestyle='dashed', label='CAD')
        return plt_asr, plt_cad
    
    def show_plots(self):
        plt.xticks(self.results_dict['param'])
        if self.param == 'c':
            plt.xlabel('tradeoff constant c') 
            plt.title(f'Compression: {self.compr}, Learning Rate: {self.constant}')
            plt.axis([0.0, 7.5, 0.0, 1.0])
        else:
            
            plt.xlabel('learning rate')
            plt.title(f'Compression: {self.compr}, Trade-Off Constant: {self.constant}')
            plt.axis([0.0, 0.01, 0.0, 1.0])
        plt.ylabel('ASR')
        plt.plot(self.results_dict['param'], self.results_dict['asr'], color='blue', label='ASR')
        plt.plot(self.results_dict['param'], self.results_dict['cad'], color='red', linestyle='dashed', label='CAD')
        plt.legend()
        plt.show()
        

class ResultsComparison:
    
    def __init__(self, results_dict_base, results_dict_target):
        self.results_dict_base = results_dict_base
        self.results_dict_target = results_dict_target
        self.param = results_dict_base.param
        self.constant = results_dict_base.constant
        self.compr = results_dict_base.compr
    
    def plot_comparison(self):
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    
        #ax1.xticks(self.results_dict_base.results_dict['param'])
        if self.param == 'c':
            ax1.set_xlabel('tradeoff constant c') 
            ax1.set_title(f'Compression: {self.compr}, Learning Rate: {self.constant}')
            ax1.axis([0.0, 7.5, 0.0, 1.0])
        else:
            
            ax1.set_xlabel('learning rate')
            ax1.set_title(f'Compression: {self.compr}, Trade-Off Constant: {self.constant}')
            ax1.axis([0.0, 0.01, 0.0, 1.0])
        ax1.set_ylabel('ASR')
        ax1.plot(self.results_dict_base.results_dict['param'], self.results_dict_base.results_dict['asr'], color='blue', label='Base')
        ax1.plot(self.results_dict_target.results_dict['param'], self.results_dict_target.results_dict['asr'], color='red', linestyle='dashed', label='Target')

        #ax2.xticks(self.results_dict_base.results_dict['param'])
        if self.param == 'c':
            ax2.set_xlabel('tradeoff constant c') 
            ax2.set_title(f'Compression: {self.compr}, Learning Rate: {self.constant}')
            ax2.axis([0.0, 7.5, 0.0, 1.0])
        else:
            
            ax2.set_xlabel('learning rate')
            ax2.set_title(f'Compression: {self.compr}, Trade-Off Constant: {self.constant}')
            ax2.axis([0.0, 0.01, 0.0, 1.0])
        ax2.set_ylabel('CAD')    
        ax2.plot(self.results_dict_base.results_dict['param'], self.results_dict_base.results_dict['cad'], color='blue', label='Base')
        ax2.plot(self.results_dict_target.results_dict['param'], self.results_dict_target.results_dict['cad'], color='red', linestyle='dashed', label='Target')
        plt.legend()
        plt.show()
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help='set path to the compression experiment you want to compare to')
    parser.add_argument('--target_dir', type=str, default='none', help='set path to the compression experiment')
    parser.add_argument('--c_or_lr', type=str, default='c', help='plot against c (c) or against learning rate (lr). Set the value in parentheses.')
    parser.add_argument('--other_constant', type=float, default=0.00001, help='constant used for the other value out of [c, lr].')
    
    
    args = parser.parse_args()
    
    if args.c_or_lr == 'c':
        param = 'c'
    else:
        param = 'lr'
    constant = args.other_constant
        
    
    compression_main_dict_base = {}
    compression_main_dict_target = {}
    
    for compr in ['70','75','80','85','90','95']:
        
        results_dict_base = ResultsDict(args.base_dir, param, constant, compr)
        results_dict_target = ResultsDict(args.target_dir, param, constant, compr)
        
        comparison = ResultsComparison(results_dict_base, results_dict_target)
        comparison.plot_comparison()
        
        
                            
    
        
        

                    
                    
                
                