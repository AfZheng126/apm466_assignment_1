import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

from assignment_1 import *

# Use Merton/KMV model to calculate the default probability of the company over time

# Default occurs when the value of the debt is higher than the value of the assets
# Equity is an option on the assets of the firm


# 
def merton():
    # The value of a company can be determined at the time debt is due
    
    # Assume the company's assets follow a geometric Brownian motion dV = mu Vdt + sigma V dz
    # Also assume no tranaction costs (including banckruptcy costs)

    # Black-Scholes gives
    # S = VN(d1) - Ke^(-rt) N(d2)

    d1 = - np.log(K * np.exp(-np.e * tau) / V) / (sigma * np.sqrt(tau))


    return 0




# Use a CreditMetrics-type model to calculate the default probability of the company over time

# plot figures
