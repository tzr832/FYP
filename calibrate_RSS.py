import numpy as np
import scipy.optimize as opt
import json
import pickle
from VSS_COS import VSSParam, VSSPricerCOS
from typing import Dict, Any

class Calibrate_RSS(VSSPricerCOS):
    def __init__(self, option_path: str = 'Data/250901.json'):
        super().__init__(VSSParam())
        with open(option_path, 'r', encoding='utf-8') as f:
            self.optiondict: Dict[str, Any] = json.load(f)

    def objective(self, x):
        param = VSSParam(*x)
        self.set_params(param)

        S0 = self.optiondict['HSI']
        # numOption = optiondict['num']
        r = self.optiondict['rf']
        error = np.empty(0)
        for key, value in self.optiondict.items():
            if not isinstance(value, dict):
                continue
            q = 0
            tau = value['tau']
            # self.n = min(126, int(tau * 63))
            # self.n = 32
            self.n = max(32, int(tau * 63))
            # self.n = int(tau * 63)
            # self.n = 32 if tau < 1 else (63 if tau < 3 else 126)
            
            modelPrice = self.price(S0, value['strike'], r, q, tau)

            error_call = (modelPrice['call'] - value['price']['call']) ** 2
            error_put = (modelPrice['put'] - value['price']['put']) ** 2
            error = np.concatenate([error, error_call, error_put])
        
        loss = np.sqrt(error.mean())
        print(f"Current loss: {loss}, Current params: {x}")
        return loss

    def calibrate(self):        
        try:
            result = opt.differential_evolution(self.objective,
                                                bounds=[(-0.01, 0.01), (-0.25, 0.25), (-0.99999999, 0.99999999), (-0.1, 0.1), (-0.2, 0.2), (1e-8, 0.99999999)],
                                                workers=2,
                                                disp=True
                                                )
            print(result)
            with open(f'opt_result.pkl', 'wb') as f:
                pickle.dump(result, f)
            print("Optimization Done!")
        except Exception as e:
            print(f"Optimization stopped due to {e}")

def main():
    pricer = Calibrate_RSS()
    pricer.calibrate()
    

def test():
    pricer = Calibrate_RSS()
    pricer.set_params(VSSParam(kappa=0.00842271,
                               nu=0.12275754,
                               rho=0.84508075,
                               X_0=-0.02704256,
                               theta=0.16772171,
                               H=0.09164193))
    x = pricer.get_params(False)
    print(pricer.objective(x))
if __name__ == "__main__":
    test()
    
