import random
import unittest
from collections import Counter
from organ_order import stratified_order


class OrderTests(unittest.TestCase):
    def test_coverage_composition_and_rng(self):
        for counts in ((5004,1820,1260),(6296,2380,1680),(2,1,1)):
            original=[(str(o),str(i)) for o,n in enumerate(counts) for i in range(n)]
            for epoch in (15,26,28):
                random.Random(2026+epoch).shuffle(original)
                state=random.getstate()
                result=stratified_order(original,epoch)
                self.assertEqual(state,random.getstate())
                self.assertEqual(result,stratified_order(original,epoch))
                self.assertEqual(Counter(result),Counter(original))
                if len(result)>16:
                    stomach=[sum(o=='2' for o,n in result[i:i+16]) for i in range(0,len(result)-15,16)]
                    self.assertGreaterEqual(min(stomach),2)
                    self.assertLessEqual(max(stomach),3)

    def test_geometry_mapping_preserves_per_image_draws(self):
        original=[(str(i%3),str(i)) for i in range(101)]
        rng=random.Random(2026); clone=random.Random(); clone.setstate(rng.getstate())
        planned={item:(clone.randrange(4),clone.randrange(2)) for item in original}
        expected={item:(rng.randrange(4),rng.randrange(2)) for item in original}
        self.assertEqual(planned,expected)
        self.assertEqual(clone.getstate(),rng.getstate())
        positions={item:i for i,item in enumerate(original)}
        for item in stratified_order(original,15):
            self.assertEqual(planned[item],expected[item])
            self.assertEqual(positions[item],original.index(item))


if __name__=='__main__': unittest.main()
