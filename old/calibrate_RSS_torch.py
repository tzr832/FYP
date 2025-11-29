import numpy as np
import torch
import json
from math import atanh, log
from torch import sigmoid

from VSS_COS_torch import VSSParamTorch, VSSPricerCOSTorch

def inv_sigmoid(y):
    return log(y / (1 - y))

class CalibrateRSS(VSSPricerCOSTorch):
    def __init__(self, dict_path = "Data/250901.json", device=None):
        super().__init__(VSSParamTorch())
        with open(f'{dict_path}', 'r') as f:
            self.option_dict: dict = json.load(f)
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
    def objective(self):
        S0 = self.option_dict['HSI']
        # numOption = optiondict['num']
        r = self.option_dict['rf']
        error = torch.empty(0)
        for value in self.option_dict.values():
            if not isinstance(value, dict):
                continue

            q = None
            tau = torch.tensor(value['tau'], dtype=torch.float64)
            self.n = max(32, int(tau * 63))
            strike = {'call': torch.tensor(value['strike']['call'], dtype=torch.float64),
                      'put':  torch.tensor(value['strike']['put'],  dtype=torch.float64)}
            
            modelPrice = self.price(S0, strike, r, q, tau)

            mkt_call = torch.tensor(value['price']['call'], dtype=torch.float64)
            mkt_put = torch.tensor(value['price']['put'], dtype=torch.float64)
            error_call = (modelPrice['call'] - mkt_call) ** 2
            error_put = (modelPrice['put'] - mkt_put) ** 2
            error = torch.concatenate([error, error_call, error_put])
        
        loss = torch.sqrt(error.mean())
        return loss

    def calibrate(self, init_param: VSSParamTorch=None, maxIter = 1000):
        if init_param == None:
            init_kappa = torch.tensor(-8.9e-5, dtype=torch.float64, requires_grad=True)
            init_nu = torch.tensor(0.176, dtype=torch.float64, requires_grad=True)
            init_rho = torch.tensor(atanh(-0.704), dtype=torch.float64, requires_grad=True)
            init_theta = torch.tensor(-0.044, dtype=torch.float64, requires_grad=True)
            init_X0 = torch.tensor(0.113, dtype=torch.float64, requires_grad=True)
            init_H = torch.tensor(inv_sigmoid(0.279), dtype=torch.float64, requires_grad=True)
        else:
            init_kappa = torch.tensor(init_param.kappa.item(), dtype=torch.float64, requires_grad=True)
            init_nu = torch.tensor(init_param.nu.item(), dtype=torch.float64, requires_grad=True)
            init_rho = torch.tensor(atanh(init_param.rho.item()), dtype=torch.float64, requires_grad=True)
            init_theta = torch.tensor(init_param.theta.item(), dtype=torch.float64, requires_grad=True)
            init_X0 = torch.tensor(init_param.X_0.item(), dtype=torch.float64, requires_grad=True)
            init_H = torch.tensor(inv_sigmoid(init_param.H.item()), dtype=torch.float64, requires_grad=True)

        optmizer = torch.optim.Adam(params=[init_kappa, init_nu, init_rho, init_theta, init_X0, init_H],
                                    lr = 0.01
                                    )
        prev_loss = torch.inf
        for epoch in range(maxIter):
            rho = torch.tanh(init_rho)
            H = torch.sigmoid(init_H)

            param = VSSParamTorch()
            param.kappa = init_kappa
            param.nu = init_nu
            param.rho = rho
            param.theta = init_theta
            param.X_0 = init_X0
            param.H = H
            self.params = param

            optmizer.zero_grad()
            loss = self.objective()
            loss.backward()
            optmizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.6f}", end=" | ")
                print(f"Params: kappa={self.params.kappa.item():.8f}, nu={self.params.nu.item():.6f}, "
                      f"rho={rho.item():.6f}, theta={self.params.theta.item():.6f}, "
                      f"X_0={self.params.X_0.item():.6f}, H={H.item():.6f}")
            
            if abs(loss - prev_loss) < 1e-3:
                print(f"Optimization finished! The optimum paramters are found within given tolerance (1e-3)")
                return {"suc": True, "loss": loss.item(), "param": param.to_dict()}
            prev_loss = loss

        print(f"Optimization finished! The optimum paramters are not found within given tolerance (1e-3)")
        return {"suc": False, "loss": loss.item(), "param": param.to_dict()}


def test():
    import time
    calibrator = CalibrateRSS(device='cuda')
    init_param = VSSParamTorch(kappa=0.00842271,
                               nu=0.12275754,
                               rho=0.84508075,
                               X_0=-0.02704256,
                               theta=0.16772171,
                               H=0.09164193)
    calibrator.params = init_param
    start = time.time()
    loss = calibrator.objective()
    print(f"forward: {time.time() - start}")
    start = time.time()
    loss.backward()
    print(f"backward: {time.time() - start}")

    print(f"Loss={loss.item():.6f}", end=" | ")
    print(f"Params: kappa={calibrator.params.kappa.item():.8f}, nu={calibrator.params.nu.item():.6f}, "
            f"rho={calibrator.params.rho.item():.6f}, theta={calibrator.params.theta.item():.6f}, "
            f"X_0={calibrator.params.X_0.item():.6f}, H={calibrator.params.H.item():.6f}")
    print(f"Grads: kappa={calibrator.params.kappa.grad.item():.8f}, nu={calibrator.params.nu.grad.item():.6f}, "
            f"rho={calibrator.params.rho.grad.item():.6f}, theta={calibrator.params.theta.grad.item():.6f}, "
            f"X_0={calibrator.params.X_0.grad.item():.6f}, H={calibrator.params.H.grad.item():.6f}")

def main():
    calibrator = CalibrateRSS(device='cpu')
    # print(calibrator.objective())
    
    init_param = VSSParamTorch(kappa=0.00842271,
                               nu=0.12275754,
                               rho=0.84508075,
                               X_0=-0.02704256,
                               theta=0.16772171,
                               H=0.09164193)

    result = calibrator.calibrate(init_param=init_param)
    with open("VSS_calibration_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f)
    print("Optimization result has been saved")


if __name__ == "__main__":
    main()