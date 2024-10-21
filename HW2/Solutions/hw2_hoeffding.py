import numpy as np

class FairCoin:
    def __init__(self, num_flips):
        self.num_flips = max(1, num_flips)
        #heads = 1, tails = 0
        self.flips = np.random.randint(0, 2, self.num_flips)
        self.freq_heads = np.average(self.flips)

class CoinFlips:
    def __init__(self, num_coins):
        #num_exp = num experiments
        self.num_coins = max(1, num_coins)
        self.coin_flips = [FairCoin(10) for x in range(self.num_coins)]
        self.head_freqs = np.array([x.freq_heads for x in self.coin_flips])
        self.first_coin_freq = self.head_freqs[0]
        self.rand_coin_freq = self.head_freqs[np.random.randint(0, self.num_coins)]
        self.min_coin_freq = self.head_freqs.min()
        
def prob12(num_exp):
    hw2_hexps = [CoinFlips(1000) for x in range(num_exp)]
    nu_mins = np.array([x.min_coin_freq for x in hw2_hexps])
    nu_firsts = np.array([x.first_coin_freq for x in hw2_hexps])
    nu_rand = np.array([x.rand_coin_freq for x in hw2_hexps])
    mins_avg = np.average(nu_mins)
    firsts_avg = np.average(nu_firsts)
    rand_avg = np.average(nu_rand)
    print("Average number of heads:")
    print("first coins of sims: %f" % firsts_avg)
    print("coin w/ min freq of heads of sims: %f" %  mins_avg)
    print("rand selection of coin of sims: %f" % rand_avg)
    
