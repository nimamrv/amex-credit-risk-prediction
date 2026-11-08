import unittest
import pandas as pd
from src.features import build_features

class TestFeatures(unittest.TestCase):
    def test_cutoff_and_customer_alignment(self):
        df=pd.DataFrame({'customer_ID':['a','a','b','b'],'S_2':['2018-02-01','2018-05-01','2018-02-01','2018-03-01'],
                         'target':[0,0,1,1],'P_2':[1.,999.,3.,5.], 'B_30':[1,0,0,1]})
        got=build_features(df).set_index('customer_ID')
        self.assertEqual(got.loc['a','P_2_max'],1.)
        self.assertEqual(got.loc['b','P_2_mean'],4.)
        self.assertEqual(got.loc['b','target'],1)
        self.assertEqual(got.loc['a','days_since_last_statement'],58)

if __name__=='__main__': unittest.main()
