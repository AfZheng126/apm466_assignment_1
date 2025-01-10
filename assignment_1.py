import pandas as pd
import numpy as np

from datetime import datetime
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

def calculate_days_since_last_coupon_payment(date):
    if date.month < 7:
        start_date = datetime(date.year, 1, 1)
    else:
        start_date = datetime(date.year, 6, 30)
    
    return (date - start_date).days

def calculate_dirty_prices(price, days_since_last_coupon_payment, coupon_rate):
    if price == "-":
        return "-"
    else:
        return days_since_last_coupon_payment / 365 * coupon_rate * 100  + price # assuming 100$ bond

def calcualte_yield_rates(yield_rates, dirty_price, coupon_rate, years_until_maturity):
    if years_until_maturity < 0.5:
        notional = 100 + coupon_rate / 2 * 100
        # print(f"notional: {notional}, {dirty_price / notional}, {years_until_maturity}, {np.log(dirty_price / notional)}")
        val = - np.log(dirty_price / notional) / years_until_maturity
        return val

    number_of_coupon_payments = int(np.floor(years_until_maturity * 2))
    # print(f"T: {years_until_maturity}")

    price = dirty_price
    # subtract the previous payments
    for i in range(number_of_coupon_payments - 1):
        t = years_until_maturity - (number_of_coupon_payments - i - 1) * 0.5
        # print(f"temp t: {t}")
        N = 100 * coupon_rate / 2

        if t in yield_rates:
            r = yield_rates[t]
        else:
            r = bootstrap_yield_rate(yield_rates, t)
        price -= N * np.exp(- r * t)

    N = 100 + 100 * coupon_rate / 2
    return - np.log(price / N) / years_until_maturity

def bootstrap_yield_rate(d, x):
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
    # print(f"x: {x}. xl: {xl}, xu: {xu}, yl: {yl}, yu: {yu}")

    # if there is no xu, then just pretend x = xl // TODO: ASK WHAT TO DO HERE
    if xu == 0:
        return yl
    else:
        # Perform linear interpolation
        interpolated_value = yl + (x - xl) * (yu - yl) / (xu - xl)
        return interpolated_value

def create_sorted_bonds_by_maturity():
    # If you want to read only a subset of sheets
    # xls = pd.ExcelFile('bond_prices.xlsx')
    # sheet_names = xls.sheet_names
    # Filter sheet names that start with 'Desired'
    # desired_sheets = [sheet for sheet in sheet_names if sheet.startswith('Desired')]
    # dfs = pd.read_excel('bond_prices.xlsx', sheet_name=desired_sheets, usecols = ['Coupon', 'Maturity Date', 'Bid', 'Ask', 'Years until Maturity'])

    dfs = pd.read_excel('bond_prices.xlsx', sheet_name=None, usecols = ['Coupon', 'Maturity Date', 'Bid', 'Ask', 'Years until Maturity'])
    
    for sheet_name, df in dfs.items():
        print(f"Processing sheet: {sheet_name}")
        # get all the required information for the bonds
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
        df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
        df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Ask'], row['Days since last coupon payment'], row['Coupon']), axis= 1)
        # print(df[['Dirty price', 'Years until Maturity']])

    return dfs

def calculate_all_yield_rates(bonds):
    all_yield_rates = {}
    for date, bond in bonds.items():
        # print(f"Processing sheet: {date}")
        # sort the bonds by years until maturity
        bond = bond.sort_values(by='Years until Maturity', ascending=True)
        # print(bond[['Dirty price', 'Years until Maturity']])

        # calcualte the yield rates
        yield_rates = {}
        # TODO: What to do if duplicate values of t give different values of r(t)
        for index, row in bond.iterrows():
            t = row['Years until Maturity']
            if row['Dirty price'] == "-":
                continue
            yield_rates[t] = calcualte_yield_rates(yield_rates, row['Dirty price'], row['Coupon'], t)
            # print(f"t: {t}, yield rate: {yield_rates[t]}, dirty price: {row['Dirty price']}, coupon rate: {row['Coupon']}")
        
        # for key in yield_rates:
        #     print(f"y: {key}, val: {yield_rates[key]}")
        all_yield_rates[date] = yield_rates
    return all_yield_rates

def plot_yield_curve(yield_rates):
    cmap = plt.get_cmap('tab10')

    for i, date in enumerate(yield_rates):
        # plot curve for each date
        years_until_maturity = list(yield_rates[date].keys())
        yield_rates_values = list(yield_rates[date].values())
        
        # get the label for the plot
        words = date.split()
        name = ' '.join(words[1:])

        plt.plot(years_until_maturity, yield_rates_values, label = name, marker='o', linestyle='-', color=cmap(i / len(yield_rates)))
    
    # add labels and title
    plt.xlabel('Years until Maturity')
    plt.ylabel('Yield')
    plt.title('0-5 Year Yield To Maturity Curve')
    plt.legend(loc='upper right')


def calculate_ytm(dirty_price, coupon_rate, years_until_maturity):
    coupon_payment = 100 * coupon_rate / 2
    if years_until_maturity < 0.5:
        notional = 100 + coupon_payment
        # print(f"notional: {notional}, {dirty_price / notional}, {years_until_maturity}, {np.log(dirty_price / notional)}")
        val = - np.log(dirty_price / notional) / years_until_maturity
        return val
    
    number_of_coupon_payments = int(np.floor(years_until_maturity * 2))
    
    # define function for continuous compounding ytm
    def equation(r):
        # consider all coupon payments
        present_value = 0
        for i in range(number_of_coupon_payments - 1):
            t = years_until_maturity - (number_of_coupon_payments - i - 1) * 0.5
            present_value += coupon_payment * np.exp(-r * t)
        present_value += (100 + coupon_payment) * np.exp(-r * years_until_maturity)
        return present_value - dirty_price
    
    # use fsolve to solve the equation of ytm for r
    ytm = fsolve(equation, 0.05) # start the search at 5%
    return ytm[0]

def calculate_ytm_curve(bonds):
    all_ytm_vals = {}
    for date, bond in bonds.items():
        # print(f"Processing sheet: {date}")
        # sort the bonds by years until maturity
        bond = bond.sort_values(by='Years until Maturity', ascending=True)

        # calculate the spot rates
        ytm_vals = {}
        for index, row in bond.iterrows():
            t = row['Years until Maturity']
            if row['Dirty price'] == "-":
                continue
            ytm_vals[t] = calculate_ytm(row['Dirty price'], row['Coupon'], t)
        all_ytm_vals[date] = ytm_vals

    return all_ytm_vals

# plot ytm curve
def plot_ytm_curve(ytm_rates):

    cmap = plt.get_cmap('tab10')

    for i, date in enumerate(ytm_rates):
        # plot curve for each date
        years_until_maturity = list(ytm_rates[date].keys())
        ytm_values = list(ytm_rates[date].values())
        
        # get the label for the plot
        words = date.split()
        name = ' '.join(words[1:])

        plt.plot(years_until_maturity, ytm_values, label = name, marker='o', linestyle='-', color=cmap(i / len(ytm_rates)))
    
    # add labels and title
    plt.xlabel('Years until Maturity')
    plt.ylabel('Yield')
    plt.title('0-5 Year YTM Curve')
    plt.legend(loc='upper right')

def plot_difference(yield_rates, ytm_rates):

    cmap = plt.get_cmap('tab10')

    for i, date in enumerate(ytm_rates):
        # plot curve for each date
        years_until_maturity = list(ytm_rates[date].keys())
        ytm_values = list(ytm_rates[date].values())
        yield_values = list(yield_rates[date].values())
        result = []
        for i in range(len(ytm_values)):
            result.append(ytm_values[i] - yield_values[i])
        
        # get the label for the plot
        words = date.split()
        name = ' '.join(words[1:])

        plt.plot(years_until_maturity, result, label = name, marker='o', linestyle='-', color=cmap(i / len(ytm_rates)))
    
    # add labels and title
    plt.xlabel('Years until Maturity')
    plt.ylabel('Yield')
    plt.title('0-5 Year YTM Curve')
    plt.legend(loc='upper right')

# get the data
sorted_bonds = create_sorted_bonds_by_maturity()
yield_rates = calculate_all_yield_rates(sorted_bonds)
ytm_rates = calculate_ytm_curve(sorted_bonds)

# plot the data

plt.figure()
plot_yield_curve(yield_rates)
plt.figure()
plot_ytm_curve(ytm_rates)
# plt.figure()
# plot_difference(yield_rates, ytm_rates)
plt.show()
