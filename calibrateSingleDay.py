import numpy as np
import scipy.optimize as opt
import json
import pickle
from VolterraSteinSteinCOS import VSSParam, VSSPricerCOS


DEBUG = True
roundcount = 0

def _loss_func(x: tuple, *args):
    kappa, nu, rho, theta, X0, H = x
    pricer, optiondict = args

    pricer.set_params(VSSParam(kappa, nu, rho, theta, X0, H))
    S0 = optiondict['HSI']
    # numOption = optiondict['num']
    r = optiondict['rf']
    error = np.empty(0)
    for key, value in optiondict.items():
        if not isinstance(value, dict):
            continue
        q = 0
        tau = value['tau']
        # pricer.n = min(126, int(tau * 63))
        # pricer.n = 32
        # pricer.n = max(32, int(tau * 63))
        # pricer.n = int(tau * 63)
        pricer.n = 32 if tau < 1 else (63 if tau < 3 else 126)
        
        modelPrice = pricer.price(S0, value['strike'], r, q, tau)

        error_call = (modelPrice['call'] - value['price']['call']) ** 2
        error_put = (modelPrice['put'] - value['price']['put']) ** 2
        error = np.concatenate([error, error_call, error_put])
    
    loss = np.sqrt(error.mean())

    if DEBUG:
        print(f"loss: {loss}, param:{x}")
    return loss



def _callback(xk, fun=None, context=None):
    global roundcount
    print(f"Iteration {roundcount} finished")
    print(f"loss: {fun}, params:{xk}")
    roundcount += 1

if __name__ == "__main__":
    pricer = VSSPricerCOS(VSSParam())
    with open('Data/250901.json', 'r', encoding='utf-8') as f:
        optiondict = json.load(f)
    if DEBUG:
        import time
        start = time.time()
        print(_loss_func(pricer.get_params(False), pricer, optiondict))
        print(f"It takes {time.time() - start}s to run loss function once.")
    # result = opt.minimize(_loss_func, 
    #                       x0=pricer.get_params(False),
    #                       bounds=[(-0.01, 0.01), (0, 0.25), (-0.99999999, 0.99999999), (-0.1, 0.1), (-0.2, 0.2), (1e-8, 0.99999999)],
    #                       args=(pricer, optiondict),
    #                       method='L-BFGS-B',
    #                       callback=_callback)

    # result = opt.differential_evolution(_loss_func,
    #                                     bounds=[(-0.01, 0.01), (0, 0.25), (-0.99999999, 0.99999999), (-0.1, 0.1), (-0.2, 0.2), (1e-8, 0.99999999)],
    #                                     # callback=_callback,
    #                                     args=(pricer, optiondict),
    #                                     disp=True,
    #                                     polish=False,
    #                                     workers=-1,
    #                                     updating='deferred'
    #                                     )
    result = opt.dual_annealing(_loss_func,
                                bounds=[(-0.01, 0.01), (0, 0.25), (-0.99999999, 0.99999999), (-0.1, 0.1), (-0.2, 0.2), (1e-8, 0.99999999)],
                                callback=_callback,
                                args=(pricer, optiondict),
                                )

    print(result)
    with open('opt_result.pkl', 'wb') as f:
        pickle.dump(result, f)
    print("Optimization Done!")
    

