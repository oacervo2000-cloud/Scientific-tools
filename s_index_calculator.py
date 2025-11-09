import numpy as np

def rotation(S,bv,tal):
    #tal = 12.8 # TGEC
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
    return Prot,rosby,np.log10(Rhk_)

def Ro(Rhk_):
    """
    Compute the Rossby number from the observational activity index S-index and the color index (B-V).
    This method was derived from Noyes et al. 1984
    """
    y = np.log10(10**(5)*Rhk_)
    Ro = logptal = 0.324 - (0.400*y) - (0.283*(y**2)) - (0.1325*(y**3))
    return Ro

def Rhk_(S,bv):
    """
    Compute the chromospheric activity index R'hk from the observational activity index S-index.
    This method was derived from Noyes et al. 1984
    """
    #bv=0.65
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
    Compute the chromospheric activity index R'hk from the observational activity index S-index.
    This method was derived from Noyes et al. 1984
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


def tau (bv):
    """
    Compute the convective turner overtime from the observational the color index (B-V).
    This method was derived from Noyes et al. 1984
    """
    x = 1 - bv
    if x>0:
        k = 1.362 - 0.166*x + 0.025*(x**2) -5.323*(x**3)
    else:
        k = 1.362 - 0.14*x
    Tau = 10**k
    return Tau

def S_index(wav,flux):
    lamh = 3933.663
    lamk = 3968.469
    lamv = 3900.000
    lamr = 4000.000
    deltalamca = 2.18/2
    deltalamcont = 20/2

    h_mask = (wav > lamh - deltalamca) & (wav < lamh + deltalamca)
    k_mask = (wav > lamk - deltalamca) & (wav < lamk + deltalamca)
    r_mask = (wav > lamr - deltalamcont) & (wav < lamr + deltalamcont)
    v_mask = (wav > lamv - deltalamcont) & (wav < lamv + deltalamcont)

    fluxh = np.sum(flux[h_mask] * (deltalamca - np.abs(wav[h_mask] - lamh)) / (deltalamca * 4))
    weighth = np.sum((deltalamca - np.abs(wav[h_mask] - lamh)) / (deltalamca * 4))

    fluxk = np.sum(flux[k_mask] * (deltalamca - np.abs(wav[k_mask] - lamk)) / (deltalamca * 4))
    weightk = np.sum((deltalamca - np.abs(wav[k_mask] - lamk)) / (deltalamca * 4))

    fluxr = np.sum(flux[r_mask])
    weightr = np.sum(r_mask)

    fluxv = np.sum(flux[v_mask])
    weightv = np.sum(v_mask)

    fluxv /= weightv
    fluxr /= weightr
    fluxh /= weighth
    fluxk /= weightk

    result = (fluxh + fluxk)/(fluxv + fluxr)
    return result
