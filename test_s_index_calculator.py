import unittest
import numpy as np
import s_index_calculator as sic

class TestSIndexCalculator(unittest.TestCase):

    def test_S_index(self):
        """
        Testa a função S_index com dados de espectro simulados.
        """
        # Cria um espectro simulado
        wav = np.linspace(3800, 4100, 3000)
        flux = np.ones_like(wav)

        # Adiciona linhas de absorção simuladas para H e K
        lamh = 3933.663
        lamk = 3968.469
        flux[(wav > lamh - 1) & (wav < lamh + 1)] = 0.5
        flux[(wav > lamk - 1) & (wav < lamk + 1)] = 0.5

        s_index = sic.S_index(wav, flux)

        # O valor esperado depende dos parâmetros de simulação, mas deve ser > 0
        self.assertGreater(s_index, 0)
        # Para este espectro simulado, o valor deve ser próximo de 0.5
        self.assertAlmostEqual(s_index, 0.5, delta=0.1)

    def test_Rhk_(self):
        """
        Testa a função Rhk_ com valores conhecidos.
        """
        # Valores de teste baseados em dados conhecidos
        S = 0.167
        bv = 0.65
        rhk_ = sic.Rhk_(S, bv)
        self.assertAlmostEqual(rhk_, 1.23e-5, delta=1e-7)

    def test_tau(self):
        """
        Testa a função tau com um valor de B-V conhecido.
        """
        bv = 0.65
        tau = sic.tau(bv)
        self.assertAlmostEqual(tau, 11.98, delta=0.1)

if __name__ == '__main__':
    unittest.main()
