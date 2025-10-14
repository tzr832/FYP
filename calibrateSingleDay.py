import numpy as np
import scipy.optimize as opt
import json
from VolterraSteinSteinCOS import VSSParam, VSSPricerCOS

def _loss_func(x: tuple, *args):
    kappa, nu, rho, theta, X0, H = x
    pricer, optiondict = args

    pricer.set_params(VSSParam(kappa, nu, rho, theta, X0, H))
    S0 = optiondict['HSI']
    # numOption = optiondict['num']
    error = np.empty(0)
    for key, value in optiondict.items():
        if not isinstance(value, dict):
            continue
        r = value['rf']
        q = 0
        tau = value['tau']
        pricer.n = max(32, int(tau / 4))
        modelPrice = pricer.price(S0, value['strike'], r, q, tau)

        error_call = (modelPrice['call'] - value['price']['call']) ** 2
        error_put = (modelPrice['put'] - value['price']['put']) ** 2
        error = np.concatenate([error, error_call, error_put])
    print(f"loss: {np.sqrt(error.mean())}, param:{x}")
    return np.sqrt(error.mean())

roundcount = 0
def _callback(xk, _):
    global roundcount
    print(f"Iteration {roundcount} finished")
    roundcount += 1

if __name__ == "__main__":
    
    pricer = VSSPricerCOS(VSSParam())
    with open('Data/250901.json', 'r', encoding='utf-8') as f:
        optiondict = json.load(f)

    # result = opt.minimize(_loss_func, 
    #                       x0=pricer.get_params(False),
    #                       bounds=[(-1., 1.), (0, 1), (-1, 1), (-1, 1), (0, 1), (0, 1)],
    #                       args=(pricer, optiondict),
    #                       method='L-BFGS-B',
    #                       callback=_callback)

    result = opt.differential_evolution(_loss_func,
                                        bounds=[(-1., 1.), (0, 1), (-1, 1), (-1, 1), (0, 1), (0, 1)],
                                        callback=_callback,
                                        args=(pricer, optiondict),
                                        workers=-1,
                                        polish=True,
                                        updating='deferred'
                                        )
    
    print("Optimization Done!")
    # import time
    # start = time.time()
    # print(_loss_func((kappa, nu, rho, theta, X0, H), pricer, optiondict))
    # print(f"It takes {time.time() - start}s to run loss function once.")

