"""Compare reduced submission kernels against the experiment implementation."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class DeliveryTests(unittest.TestCase):
    def test_initialization_loss_and_update_match(self):
        env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
        with tempfile.TemporaryDirectory() as folder:
            results=[]
            for mode,root in [('original',ROOT),('delivery',ROOT/'tools/submission_round245')]:
                output=Path(folder)/(mode+'.json')
                subprocess.run([sys.executable,str(ROOT/'tests/compare_round245_delivery_math.py'),'--mode',mode,'--root',str(root),'--output',str(output)],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
                results.append(json.loads(output.read_text()))
            self.assertEqual(results[0],results[1])

if __name__=='__main__':unittest.main()
