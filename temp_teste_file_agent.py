import unittest

class TestDivisao(unittest.TestCase):

    def setUp(self):
        pass

    def test_dividir_por_zero(self):
        try:
            self.assertEqual(5 / 0, 1)
        except ZeroDivisionError as e:
            self.assertEqual(str(e), "division by zero")

    def test_dividir_por_numero_valido(self):
        self.assertEqual(5 / 3, 1.666666666666667)

if __name__ == "__main__":
    unittest.main()