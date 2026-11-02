"""800轮DAPI域内预训练入口。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from pretrain_core import main
if __name__=='__main__':main()
