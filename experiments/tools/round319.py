"""Full-data refit and package of the Val-selected Round318 recipe."""
import argparse
import os
import sys
import traceback
import round308 as recipe
from round314 import main

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--log',action='store_true')
    args=p.parse_args(); os.chdir(recipe.ROOT)
    if args.log:
        sys.stdout=open('reports/round319.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round319.err','a',encoding='utf-8',buffering=1)
    try: main(args.smoke,319)
    except Exception:
        traceback.print_exc(); sys.exit(1)
