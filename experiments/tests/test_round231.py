import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round231_droppath import command
from run_round204_cd45_cv import training_command

class DropPathScreenTests(unittest.TestCase):
    def test_only_drop_path_and_artifact_sources_change(self):
        for name in ['fold0','fold1']:
            f={'name':name,'val_rois':['ROI005']}
            for stage in (1,2):
                expected,_=training_command(f,stage,14)
                expected[expected.index('--output-dir')+1]=f'runs/round231_{name}_stage{stage}'
                if stage==1:expected[expected.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/{'round207' if name=='fold0' else 'round216'}_{name}_fcmae800/final_encoder.pt"
                else:expected[expected.index('--init-checkpoint')+1]=f'runs/round231_{name}_stage1/epochs/epoch_014.pt'
                expected+=['--encoder-drop-path-rate','0.2']
                self.assertEqual(command(f,stage),expected)

if __name__=='__main__':unittest.main()
