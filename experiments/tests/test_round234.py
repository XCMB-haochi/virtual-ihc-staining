import sys
import unittest
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from ihc_unext2 import UNeXt2VirtualStainer,UNeXt2LocalConditionVirtualStainer,unext2_local_condition_model_specification
from ihc_spatial import SPATIAL_CONDITION_FEATURE_NAMES, spatial_thumbnail_conditioning_specification
from ihc_torch import build_model_from_checkpoint
from run_round234_local import command
from run_round204_cd45_cv import training_command

class LocalConditionTests(unittest.TestCase):
    def test_recipe_only_changes_architecture_and_paths(self):
        f={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
        for stage in (1,2):
            expected,_=training_command(f,stage,14)
            expected[expected.index('--model-arch')+1]='unext2-fcmae-tiny-local-condition'
            expected[expected.index('--output-dir')+1]=f'runs/round234_fold3_stage{stage}'
            if stage==1:expected[expected.index('--fcmae-encoder-init-checkpoint')+1]='runs/round207_fold3_fcmae800/final_encoder.pt'
            else:expected[expected.index('--init-checkpoint')+1]='runs/round234_fold3_stage1/epochs/epoch_014.pt'
            self.assertEqual(command(f,stage),expected)

    def test_exact_mask_and_saved_model_behavior(self):
        torch.set_num_threads(2);torch.manual_seed(2026)
        base=UNeXt2VirtualStainer(out_channels=4,condition_mode='spatial-thumbnail-film')
        torch.manual_seed(2026)
        model=UNeXt2LocalConditionVirtualStainer(out_channels=4)
        for key,value in base.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        spec=unext2_local_condition_model_specification()
        self.assertEqual(list(SPATIAL_CONDITION_FEATURE_NAMES[9:20]),spec['masked_condition_features'])
        x=torch.rand(1,1,32,32);condition=torch.rand(1,model.condition_dimension);copy=condition.clone()
        masked=model._validate_condition(x,condition)
        self.assertTrue(torch.equal(condition,copy))
        self.assertEqual(float(masked[:,9:20].abs().sum()),0)
        self.assertTrue(torch.equal(masked[:,:9],condition[:,:9]))
        self.assertTrue(torch.equal(masked[:,20:],condition[:,20:]))
        with torch.no_grad():
            for p in model.condition_film.parameters():p.add_(.001)
        ck={'model':model.state_dict(),'markers':['CD68','CD45RO','HLA-DR','Vimentin'],'model_spec':spec,
            'conditioning':spatial_thumbnail_conditioning_specification(),
            'args':{'model_arch':spec['architecture'],'condition_mode':'spatial-thumbnail-film'}}
        restored=build_model_from_checkpoint(ck,torch.device('cpu'))
        changed=condition.clone();changed[:,9:20]+=100
        with torch.no_grad():
            torch.testing.assert_close(model(x,condition),model(x,changed),rtol=0,atol=0)
            torch.testing.assert_close(model(x,condition),restored(x,condition),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
