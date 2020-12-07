import unittest


class TestExample(unittest.TestCase):

    # Returns True or False.
    def test(self):
        print("测试代码开始")
        self.assertTrue(1 == 1)
