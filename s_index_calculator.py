import numpy as np

def rotation(S, bv, tal):
    """
    Calcula o período de rotação (Prot), o número de Rossby (rosby) e o logaritmo do
    índice de atividade cromosférica (logRhk_) a partir do índice S, do índice de cor (B-V)
    e do tempo de turnover convectivo (tal).
    """
    logCcf = (1.13*(bv**3) - 3.91*(bv**2) + 2.84*(bv) - 0.47)
    x = 0.63 - bv
    deltalogC = (0.135*x) - (0.814*(x**2)) + (6.03*(x**3))
    logCcf_ = (logCcf + deltalogC)
    Ccf_ = 10**logCcf_
    Rhk = 1.340*(10**(-4))*Ccf_*S
    logRphot = (-4.02 - 1.40*(bv))
    Rphot = 10**logRphot
    Rhk_ = Rhk - Rphot
    y = np.log10(10**(5)*Rhk_)
    logptal = 0.324 - (0.400*y) - (0.283*(y**2)) - (0.1325*(y**3))
    Prot = tal * 10**(logptal)
    rosby = Prot / tal
    return Prot, rosby, np.log10(Rhk_)

def Ro(Rhk_):
    """
    Calcula o número de Rossby a partir do índice de atividade cromosférica R'hk.
    Este método foi derivado de Noyes et al. 1984.
    """
    y = np.log10(10**(5)*Rhk_)
    Ro = 0.324 - (0.400*y) - (0.283*(y**2)) - (0.1325*(y**3))
    return Ro

def Rhk_(S, bv):
    """
    Calcula o índice de atividade cromosférica R'hk a partir do índice S observacional e do
    índice de cor (B-V).
    Este método foi derivado de Noyes et al. 1984.
    """
    logCcf = (1.13*(bv**3) - 3.91*(bv**2) + 2.84*(bv) - 0.47)
    x = 0.63 - bv
    deltalogC = (0.135*x) - (0.814*(x**2)) + (6.03*(x**3))
    logCcf_ = (logCcf + deltalogC)
    Ccf_ = 10**logCcf_
    Rhk = 1.340*(10**(-4))*Ccf_*S
    logRphot = (-4.02 - 1.40*(bv))
    Rphot = 10**logRphot
    Rhk_ = Rhk - Rphot
    return Rhk_

def Rhk(S):
    """
    Calcula o índice de atividade cromosférica Rhk a partir do índice S, assumindo um
    índice de cor (B-V) fixo de 0.65.
    """
    bv=0.65
    logCcf = (1.13*(bv**3) - 3.91*(bv**2) + 2.84*(bv) - 0.47)
    x = 0.63 - bv
    deltalogC = (0.135*x) - (0.814*(x**2)) + (6.03*(x**3))
    logCcf_ = (logCcf + deltalogC)
    Ccf_ = 10**logCcf_
    Rhk = 1.340*(10**(-4))*Ccf_*S
    return Rhk

def Rphot(S):
    """
    Calcula a contribuição fotosférica Rphot para o índice de atividade, assumindo um
    índice de cor (B-V) fixo de 0.65.
    """
    bv=0.65
    logCcf = (1.13*(bv**3) - 3.91*(bv**2) + 2.84*(bv) - 0.47)
    x = 0.63 - bv
    deltalogC = (0.135*x) - (0.814*(x**2)) + (6.03*(x**3))
    logCcf_ = (logCcf + deltalogC)
    Ccf_ = 10**logCcf_
    Rhk = 1.340*(10**(-4))*Ccf_*S
    logRphot = (-4.02 - 1.40*(bv))
    Rphot = 10**logRphot
    return Rphot

def tau(bv):
    """
    Calcula o tempo de turnover convectivo (tau) a partir do índice de cor (B-V).
    Este método foi derivado de Noyes et al. 1984.
    """
    x = 1 - bv
    if x > 0:
        k = 1.362 - 0.166*x + 0.025*(x**2) - 5.323*(x**3)
    else:
        k = 1.362 - 0.14*x
    Tau = 10**k
    return Tau

def S_index(wav, flux):
    """
    Calcula o índice de atividade S a partir do espectro de uma estrela.

    O cálculo é baseado na integração do fluxo em janelas espectrais específicas
    nas linhas de Ca II H & K, e em regiões de contínuo próximas (V e R).

    Parâmetros:
    wav (array): Array com os comprimentos de onda do espectro.
    flux (array): Array com os fluxos correspondentes do espectro.

    Retorna:
    float: O valor do índice S calculado.
    """
    # Define os comprimentos de onda centrais para as linhas H e K e para as
    # regiões de contínuo V e R.
    lamh = 3933.663
    lamk = 3968.469
    lamv = 3900.000
    lamr = 4000.000

    # Define as larguras das janelas de integração para as linhas e o contínuo.
    deltalamca = 2.18 / 2
    deltalamcont = 20 / 2

    # Cria máscaras booleanas para selecionar as regiões de interesse no espectro.
    h_mask = (wav > lamh - deltalamca) & (wav < lamh + deltalamca)
    k_mask = (wav > lamk - deltalamca) & (wav < lamk + deltalamca)
    r_mask = (wav > lamr - deltalamcont) & (wav < lamr + deltalamcont)
    v_mask = (wav > lamv - deltalamcont) & (wav < lamv + deltalamcont)

    # Calcula o fluxo integrado ponderado para as linhas H e K.
    fluxh = np.sum(flux[h_mask] * (deltalamca - np.abs(wav[h_mask] - lamh)) / (deltalamca * 4))
    weighth = np.sum((deltalamca - np.abs(wav[h_mask] - lamh)) / (deltalamca * 4))

    fluxk = np.sum(flux[k_mask] * (deltalamca - np.abs(wav[k_mask] - lamk)) / (deltalamca * 4))
    weightk = np.sum((deltalamca - np.abs(wav[k_mask] - lamk)) / (deltalamca * 4))

    # Calcula o fluxo médio nas regiões de contínuo R e V.
    fluxr = np.sum(flux[r_mask])
    weightr = np.sum(r_mask)

    fluxv = np.sum(flux[v_mask])
    weightv = np.sum(v_mask)

    # Normaliza os fluxos pelos seus respectivos pesos (número de pontos).
    fluxv /= weightv
    fluxr /= weightr
    fluxh /= weighth
    fluxk /= weightk

    # Calcula o índice S como a razão entre o fluxo nas linhas e no contínuo.
    result = (fluxh + fluxk) / (fluxv + fluxr)
    return result
