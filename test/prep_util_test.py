import unittest

import prep.dataset


class PrepTrainDataFromTextTest(unittest.TestCase):
    def test_exchange_data_simple(self):
        self.assertEqual(prep.dataset.exchange_data(None, None, None, n1=1000, n2=1000), (None, None))
        self.assertEqual(prep.dataset.exchange_data([], [], None, n1=1000, n2=1000), ([], []))
        self.assertEqual(prep.dataset.exchange_data([], [], "dodo", n1=1000, n2=1000), ([], []))
        self.assertEqual(prep.dataset.exchange_data([[1, 2, "dodo"], [3, 4, "bird"]], [], "dodo", n1=1000, n2=1000),
                         ([[1, 2, "dodo"], [3, 4, "bird"]], [[3, 4, "dodo"]]))

if __name__ == '__main__':
    unittest.main()
