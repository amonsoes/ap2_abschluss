from src.adversarial.iqa.frqa import FRQA

class FRQAExperiment:

    def __init__(self, dataset, run_name):
        self.dataset = dataset
        self.frqa = FRQA(run_name)
    
    def __call__(self):
        pass