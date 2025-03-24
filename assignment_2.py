import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

# Code for spot rates from assignment 1

def calculate_days_since_last_coupon_payment(date):
    if date.month < 3: # assume all coupons mature on March 1 or Sept 1
        start_date = datetime(date.year - 1, 9, 1)
    elif date.month > 9:
        start_date = datetime(date.year, 9, 1)
    else: 
        start_date = datetime(date.year, 3, 1)
    
    return (date - start_date).days

def calculate_dirty_prices(price, days_since_last_coupon_payment, coupon_rate):
    if price == "-":
        return "-"
    else:
        return days_since_last_coupon_payment / 365 * coupon_rate * 100  + price # assuming 100$ bond

def create_sorted_bonds_by_maturity(file_name, sheet_name):
    # If you want to read only a subset of sheets
    # xls = pd.ExcelFile('bond_prices.xlsx')
    # sheet_names = xls.sheet_names
    # Filter sheet names that start with 'Desired'
    # desired_sheets = [sheet for sheet in sheet_names if sheet.startswith('Desired')]
    # dfs = pd.read_excel('bond_prices.xlsx', sheet_name=desired_sheets, usecols = ['Coupon', 'Maturity Date', 'Bid', 'Ask', 'Years until Maturity'])

    df = pd.read_excel(file_name, sheet_name=sheet_name, usecols = ['Coupon', 'Maturity Date', 'Price', 'Years until Maturity'])
    
    print(f"Processing sheet: {sheet_name}")
   # get all the required information for the bonds
    df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
    df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
    df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Price'], row['Days since last coupon payment'], row['Coupon']), axis= 1)
    df['Years until Maturity'] = df['Years until Maturity'].round(4)
    print(df[['Dirty price', 'Years until Maturity']])

    return df

def calcualte_spot_rates(spot_rates, dirty_price, coupon_rate, years_until_maturity):
    if years_until_maturity < 0.5:
        notional = 100 + coupon_rate / 2 * 100
        # print(f"notional: {notional}, {dirty_price / notional}, {years_until_maturity}, {np.log(dirty_price / notional)}")
        val = - np.log(dirty_price / notional) / years_until_maturity
        return val

    number_of_coupon_payments = int(np.floor(years_until_maturity * 2))
    # print(f"T: {years_until_maturity}, number of coupon payments: {number_of_coupon_payments}")

    price = dirty_price
    # subtract the previous payments
    for i in range(number_of_coupon_payments):
        t = round(years_until_maturity - (number_of_coupon_payments - i) * 0.5, 4) # due to numerical errors
        # print(f"i: {i}, temp t: {t}")
        N = 100 * coupon_rate / 2 # nominal

        if t in spot_rates:
            r = spot_rates[t]
        else:
            r = interpolate_rate(spot_rates, t)
        price -= N * np.exp(- r * t)

    N = 100 + 100 * coupon_rate / 2
    return - np.log(price / N) / years_until_maturity

def interpolate_rate(d, x):
    # sort the dictionary by keys
    sorted_keys = sorted(d.keys())
    # Find the two closest keys to 'x' (one smaller and one larger)
    xl = 0
    xu = 0
    
    # print(f"finding x = {x}")
    for key in sorted_keys:
        val = d[key]
        if xl == 0:
            xl = key
            yl = val
        if key > x:
            xu = key
            yu = val
            break
        else:
            xl = key
            yl = val
    # print(f"x: {x}. xl: {xl}, xu: {xu}")

    # if there is no xu, then just pretend x = xl // TODO: ASK WHAT TO DO HERE
    if xu == 0:
        return yl
    else:
        # Perform linear interpolation
        interpolated_value = yl + (x - xl) * (yu - yl) / (xu - xl)
        return interpolated_value

def calculate_all_spot_rates(bond):

    # print(f"Processing sheet: {bonds}")
    # sort the bonds by years until maturity
    bond = bond.sort_values(by='Years until Maturity', ascending=True)
    # print(bond[['Dirty price', 'Years until Maturity']])

    # calcualte the yield rates
    spot_rates = {}
    # TODO: What to do if duplicate values of t give different values of r(t)
    for index, row in bond.iterrows():
        t = row['Years until Maturity']
        if row['Dirty price'] == "-":
            continue
        spot_rates[t] = calcualte_spot_rates(spot_rates, row['Dirty price'], row['Coupon'], t)
        # print(f"t: {t}, yield rate: {yield_rates[t]}, dirty price: {row['Dirty price']}, coupon rate: {row['Coupon']}")
    return spot_rates



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

    # d1 = - np.log(K * np.exp(-np.e * tau) / V) / (sigma * np.sqrt(tau))


    # From company financials, obtain the values for 
    # book value of assets (V)
    # book value of liabilities
    # number of shares outstanding

    # Calculate stock volatility
    # Get the share price
    # Get the rate of return (interest rates from government bonds - yield rate)

    # Equity = book value of assets - book values of liabilities
    # Future debt (exercise price) = book value of liabilities * interest rates (K)

    # market value = number of shares * share price

    # now guess that the asset volatility is stock volatility
    # Now calculate via Black Scholes the values of 
    #       option price (S)
    #       Delta
    #       fixed point = sigma_s * S / (V * Delta)
    # Change the asset volatility to equal to fixed point

    # iterate to converge to the correct asset volatility

    return 0




# Use a CreditMetrics-type model to calculate the default probability of the company over time

def calculateSolvencyProbability(bankBonds, comparyBonds):
    
    bankYears = sorted(bankBonds.keys())
    # print(bankYears)
    bankOneYearSpotRate = bankBonds[bankYears[1]]

    companyYears = sorted(comparyBonds.keys())
    # print(companyYears)
    companyOneYearSpotRate = comparyBonds[companyYears[1]]
    
    creditSpread = companyOneYearSpotRate - bankOneYearSpotRate

    # q = (e^(-h) - R) / (1 - R)
    # assume recovery rate is 50%
    R = 0.5
    solvencyProbability = (np.exp(-creditSpread) - R) / (1 - R)

    return solvencyProbability

def creditMetric(bankBonds, companyBonds):
    # Calculate spot rates
    bankSpotRates = calculate_all_spot_rates(bankBonds)
    # print("BOC")
    # for key in bankSpotRates:
    #     print(f"y: {key}, val: {bankSpotRates[key]}")
    companySpotRates = calculate_all_spot_rates(companyBonds)
    # print("TD")
    # for key in companySpotRates:
    #     print(f"y: {key}, val: {companySpotRates[key]}")

    # Calculate the credit spread for a 1 year spot rate
    probabilityOfSolency = calculateSolvencyProbability(bankSpotRates, companySpotRates)
    # print(f"probability of not defaulting: {probabilityOfSolency}")

    # Calculate probability of default in 1-10 years
    defaultProbability = []

    for i in range(1, 11):
        # print(i)
        # print(defaultProbability)
        # print((1 - probabilityOfSolency) * probabilityOfSolency**(i - 1))
        # probability = prob(solvent) * (1 - q) + prob(default) * 1
        if i == 1:
            probability = (1 - probabilityOfSolency)
        else:
            probability = (1 - probabilityOfSolency) * probabilityOfSolency**(i - 1) + defaultProbability[i-2]/100

        defaultProbability.append(probability * 100)

    # print(defaultProbability)
    return defaultProbability



# read data

bankOfCanadaBonds = create_sorted_bonds_by_maturity('bonds_data.xlsx', 'Canada bond data')
tDBonds = create_sorted_bonds_by_maturity('bonds_data.xlsx', 'TD bond data')



creditMetricModelProbabilityOfDefault = creditMetric(bankOfCanadaBonds, tDBonds)


# plot figures
