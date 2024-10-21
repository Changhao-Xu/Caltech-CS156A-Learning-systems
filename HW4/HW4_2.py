import numpy as np
import matplotlib.pyplot as plt
import math

def vc_eps(delta, N, dvc):
    return math.sqrt((8.0/N)*((dvc*math.log(2.0*N))-math.log(delta/4.0)))

def rademacher_eps(delta, N, dvc):
    cur_eps = math.sqrt((2.0/N)*(math.log(2.0*N)+(dvc*math.log(N)))) + math.sqrt((2.0/N) * math.log(1.0/delta)) + (1.0/N)
    return cur_eps

eps_range = np.arange(0.0,10, 0.0001)

def parrondo_eps(delta, N, dvc):
    last_true = False
    last_eps = eps_range[0]
    for cur_eps in eps_range:
        cur_rightside = (1.0/N) * ((2.0*cur_eps) + math.log(6.0/delta) + (dvc*math.log(2.0*N)))
        cur_rightside = math.sqrt(cur_rightside)
        cur_true = cur_eps <= cur_rightside
        if cur_true == False and last_true == True:
            break
        elif cur_true == True:
            last_true = True
            last_eps = cur_eps
    return last_eps

def devroye_eps(delta, N, dvc):
    last_true = False
    last_eps = eps_range[0]
    for cur_eps in eps_range:
        cur_rightside = (1.0/(2.0*N))*(((4.0*cur_eps)*(1.0+cur_eps)) + math.log(4.0/delta) + ((2.0*dvc)*math.log(N)))
        cur_rightside = math.sqrt(cur_rightside)
        cur_true = cur_eps <= cur_rightside
        if cur_true == False and last_true == True:
            break
        elif cur_true == True:
            last_true = True
            last_eps = cur_eps
    return last_eps

def main():
	vc_func = np.vectorize(vc_eps)

	rademacher_func = np.vectorize(rademacher_eps)

	parrondo_func = np.vectorize(parrondo_eps)

	devroye_func = np.vectorize(devroye_eps)

	prob2 = {}
	prob2["N"] = np.arange(1,10,1)
	prob2["vc"] = vc_func(0.05, prob2["N"], 50)
	prob2["rad"] = rademacher_func(0.05, prob2["N"], 50)
	prob2["par"] = parrondo_func(0.05, prob2["N"], 50)
	prob2["dev"] = devroye_func(0.05,prob2["N"],50)

	prob2["plot"] = plt.figure(figsize=(15,10), dpi=80)
	prob2["ax"] = prob2["plot"].add_subplot(111)
	prob2["ax"].set_title("Generalization Bounds")
	prob2["ax"].set_xlabel("N")
	prob2["ax"].set_ylabel("epsilon")
	prob2["ax"].set_yscale("log", basey=10)

	prob2["ax"].plot(prob2["N"], prob2["vc"], label="VC")
	prob2["ax"].plot(prob2["N"], prob2["rad"], label="Rademacher")
	prob2["ax"].plot(prob2["N"], prob2["par"], label="Parrondo/Van Den Broek")
	prob2["ax"].plot(prob2["N"], prob2["dev"], label = "Devroye")

	prob2["ax"].legend(loc='upper left', frameon=False)

	plt.show()

if __name__== "__main__":
	main()