"""Deterministic proportional interleaving, without replacement or extra images."""
import random


def stratified_order(original, epoch):
    groups={}
    for item in original:
        groups.setdefault(item[0],[]).append(item)
    # Preserve each organ's already-randomized internal order; spread its items
    # across the epoch, then randomize positions within each accumulation group.
    ranked=sorted(((2*i+1)/(2*len(items)),organ,i,item)
                  for organ,items in groups.items() for i,item in enumerate(items))
    ordered=[row[-1] for row in ranked]
    rng=random.Random(2026+epoch)
    for offset in range(0,len(ordered),16):
        group=ordered[offset:offset+16]; rng.shuffle(group)
        ordered[offset:offset+16]=group
    return ordered
