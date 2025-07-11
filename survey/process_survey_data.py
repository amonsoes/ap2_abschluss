import csv

class SurveyProcessor:
    
    def __init__(self, path):
        self.n = 0
        self.fgsm_n = 0
        self.bim_n = 0
        self.pgd_n = 0
        self.vmifgsm_n = 0
        self.vanilla_dict = {
            'fgsm' : 0,
            'bim' : 0,
            'pgd' : 0,
            'vmifgsm' : 0,
        }
        self.hpf_dict = {
            'fgsm' : 0,
            'bim' : 0,
            'pgd' : 0,
            'vmifgsm' : 0,
        }
        self.undecided_dict = {
            'fgsm' : 0,
            'bim' : 0,
            'pgd' : 0,
            'vmifgsm' : 0,
        }
        self.process_data(path)
    
    def get_results(self):
        print('Survey Question: Which image visually contained less noise?\n\n')
        print('=====================SURVEY RESULTS====================\n\n')
        print('FGSM:\n')
        print(f'\tVANILLA : {self.vanilla_dict["fgsm"] / self.fgsm_n}\n')
        print(f'\tHPF : {self.hpf_dict["fgsm"] / self.fgsm_n}\n')
        print(f'\tUNDECIDED : {self.undecided_dict["fgsm"] / self.fgsm_n}\n')
        print('=======================================================\n\n')
        print('BIM:\n')
        print(f'\tVANILLA : {self.vanilla_dict["bim"] / self.bim_n}\n')
        print(f'\tHPF : {self.hpf_dict["bim"] / self.bim_n}\n')
        print(f'\tUNDECIDED : {self.undecided_dict["bim"] / self.bim_n}\n')
        print('=======================================================\n\n')
        print('VMIFGSM:\n')
        print(f'\tVANILLA : {self.vanilla_dict["vmifgsm"] / self.vmifgsm_n}\n')
        print(f'\tHPF : {self.hpf_dict["vmifgsm"] / self.vmifgsm_n}\n')
        print(f'\tUNDECIDED : {self.undecided_dict["vmifgsm"] / self.vmifgsm_n}\n')
        print('=======================================================\n\n')
        print(f'Number of participants: {self.n}')
        
    def process_data(self, path):
        with open(path, 'r') as data_file:
            data_obj = csv.reader(data_file)
            next(data_obj)
            for line in data_obj:
                self.n += 1
                results = line[7:15]
                for e, answer in enumerate(results):
                    self.process_answer(e, answer)
    
    def process_answer(self, enum, answer):
        if answer == '3':
            if enum <= 0:
                self.undecided_dict['fgsm'] += 1
                self.fgsm_n += 1
            if enum <= 3:
                self.undecided_dict['bim'] += 1
                self.bim_n += 1
            else:
                self.undecided_dict['vmifgsm'] += 1
                self.vmifgsm_n += 1
        elif answer == '-9':
            pass
        else:
            if enum % 2 == 0:
                hpf = '2'
                vanilla = '1'
            else:
                hpf = '1'
                vanilla = '2'
            
            if answer == hpf:
                if enum <= 0:
                    self.hpf_dict['fgsm'] += 1
                    self.fgsm_n += 1
                if enum <= 3:
                    self.hpf_dict['bim'] += 1
                    self.bim_n += 1
                else:
                    self.hpf_dict['vmifgsm'] += 1
                    self.vmifgsm_n += 1
            elif answer == vanilla:
                if enum <= 0:
                    self.vanilla_dict['fgsm'] += 1
                    self.fgsm_n += 1
                if enum <= 3:
                    self.vanilla_dict['bim'] += 1
                    self.bim_n += 1
                else:
                    self.vanilla_dict['vmifgsm'] += 1
                    self.vmifgsm_n += 1
            
                
                
                
    
    

if __name__ == '__main__':
    
    """parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data.csv', help='provide the path to the survey data')
    args = parser.parse_args()"""
    
    data_path = './survey/data.csv'
    
    survey_processor = SurveyProcessor(data_path)
    survey_processor.get_results()
    
    
    