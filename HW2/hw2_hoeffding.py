import numpy as np

class EachCoin:
    def __init__(self, nflips):
        self.flips = np.random.randint(0, 2, nflips)
        self.heads = np.average(self.flips) #suppose head = 1

class CoinFlips:
    def __init__(self, ncoins):
        self.results = [EachCoin(10) for x in range(ncoins)]
        self.heads = np.array([x.heads for x in self.results])
        self.first = self.heads[0]
        self.rand = self.heads[np.random.randint(0, ncoins)]
        self.min = self.heads.min()
        
def Hoeffding(coins,times):
    results = [CoinFlips(coins) for x in range(times)]
    mins = np.array([x.min for x in results])
    firsts = np.array([x.first for x in results])
    rands = np.array([x.rand for x in results])
    minimum = np.average(mins)
    first = np.average(firsts)
    rand = np.average(rands)
    print(f"first coin: {first}")
    print(f"min freq: {minimum}")
    print(f"rand coin: {rand}")
    
def main():
    Hoeffding(1000,100000)

if __name__== "__main__":
    main()