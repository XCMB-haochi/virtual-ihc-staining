import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round219_texture import command

class TextureTests(unittest.TestCase):
 def test_only_condition_mode_and_paths_change(self):
  f={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
  from run_round204_cd45_cv import training_command
  for s in (1,2):
   base,_=training_command(f,s,14);cand=command(f,s)
   expected=[x for x in base];expected[expected.index('--condition-mode')+1]='spatial-thumbnail-texture-film';expected[expected.index('--output-dir')+1]=f'runs/round219_fold3_stage{s}'
   if s==1:expected[expected.index('--fcmae-encoder-init-checkpoint')+1]='runs/round207_fold3_fcmae800/final_encoder.pt'
   else:expected[expected.index('--init-checkpoint')+1]='runs/round219_fold3_stage1/epochs/epoch_014.pt'
   self.assertEqual(cand,expected)

if __name__=='__main__':unittest.main()
