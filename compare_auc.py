import argparse

from src.adversarial.auc_test import AUCComparison

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_dir_a', type=str, help='set name of attack dir x')
    parser.add_argument('--attack_dir_b', type=str, help='set name of attack dir y')
    parser.add_argument('--comparison_type', type=str, default='eps', help='set based on what dims AUC will be calculated')
    parser.add_argument('--comparison_basis', type=float, default=70., help='set the basis that will be used to gather the data for AUC')
    parser.add_argument('--save_figure', type=lambda x: x in ['true', 'True', '1', 'yes'], default=True, help='saves figure in cwd')
    args = parser.parse_args()
    
    auc_comparison = AUCComparison(args.attack_dir_a, args.attack_dir_b, args.comparison_type, args.comparison_basis)
    auc_comparison.process_dirs(args.save_figure)
    
    
    
    