import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round137_colon import C,SCOPES,plan,pretrain_command,student_command

class CompleteColonTests(unittest.TestCase):
    def test_graph_has_all_scopes_and_no_cycles(self):
        tasks=plan();ids={t['id'] for t in tasks};self.assertEqual(len(ids),len(tasks))
        self.assertEqual(sum(t['kind']=='dapi' for t in tasks),6)
        self.assertEqual(sum(t['kind']=='teacher' for t in tasks),6)
        self.assertEqual(sum(t['kind']=='student' for t in tasks),24)
        self.assertEqual(sum(t['kind']=='evaluate' for t in tasks),5)
        done=set()
        while len(done)<len(tasks):
            ready={t['id'] for t in tasks if set(t['deps'])<=done}-done
            self.assertTrue(ready);done|=ready
        for arm in C['arms']:
            full=next(t for t in tasks if t['id']==f'full.{arm}.s2')
            self.assertIn('cache',full['deps'])
        cache=next(t for t in tasks if t['id']=='cache');self.assertEqual(len(cache['deps']),5)

    def test_every_new_training_uses_2026_and_colon_only(self):
        for scope in SCOPES:
            for kind in ('dapi','teacher'):
                cmd=pretrain_command(scope,kind)
                self.assertEqual(cmd[2],C['dataset']);self.assertEqual(cmd[cmd.index('--seed')+1],'2026')
                self.assertEqual(cmd[cmd.index('--encoder-init-checkpoint')+1],C['public_checkpoint'])
                self.assertEqual(cmd[cmd.index('--val-rois')+1],','.join(SCOPES[scope]))
            for stage in (1,2):
                cmds=[student_command(scope,arm,stage,'testsha') for arm in C['arms']]
                for cmd in cmds:
                    self.assertEqual(cmd[cmd.index('--seed')+1],'2026')
                    self.assertEqual(cmd[cmd.index('--batch-size')+1],'16')
                    self.assertEqual(cmd[cmd.index('--gradient-accumulation-steps')+1],'1')
                    self.assertEqual('--train-all' in cmd,scope=='full')
                    self.assertEqual('--distillation-target-dir' in cmd,scope=='full' and stage==2)
                    self.assertEqual(cmd[cmd.index('--val-rois')+1],','.join(SCOPES[scope]))
                # Match commands after replacing only arm paths and decay.
                normalized=[]
                for cmd in cmds:
                    cmd=[x.replace('stage_decay','control') for x in cmd]
                    cmd[cmd.index('--encoder-stage-decay')+1]='1.0';normalized.append(cmd)
                self.assertEqual(*normalized)

    def test_pretraining_fold_boundaries(self):
        all_rois=[r for s,rois in SCOPES.items() if s!='full' for r in rois]
        self.assertEqual(len(all_rois),len(set(all_rois)));self.assertEqual(len(all_rois),25)
        for scope in SCOPES:
            cmd=pretrain_command(scope,'dapi')
            self.assertEqual(cmd[cmd.index('--epochs')+1],'630' if scope=='full' else '800')
            self.assertEqual(cmd[cmd.index('--data-scope')+1],'all-train' if scope=='full' else 'fold-train')
        self.assertEqual(C['submission_offset'],[0,0])

if __name__=='__main__':unittest.main()
