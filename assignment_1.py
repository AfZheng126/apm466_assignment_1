import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math

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

def calculate_yield_rates_for_zero_coupon_bonds(dirty_price, coupon_rate, years_until_maturity):
    notional = 100 + coupon_rate / 2 * 100
    # print(f"notional: {notional}, {dirty_price / notional}, {years_until_maturity}, {np.log(dirty_price / notional)}")
    val = - np.log(dirty_price / notional) / years_until_maturity
    return val

def calcualte_yield_rates(yield_rates, dirty_price, coupon_rate, years_until_maturity):
    number_of_coupon_payments = math.floor(years_until_maturity * 2)
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
    dfs = pd.read_excel('bond_prices.xlsx', sheet_name=None, usecols = ['Coupon', 'Maturity Date', 'Bid', 'Ask', 'Years until Maturity'])
    jan6 = dfs['January 6']
    
    # get all the required information for the bonds
    jan6['Maturity Date'] = pd.to_datetime(jan6['Maturity Date'])
    jan6['Month'] = jan6['Maturity Date'].dt.month
    jan6['Days since last coupon payment'] = jan6['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
    jan6['Dirty price'] = jan6.apply(lambda row: calculate_dirty_prices(row['Ask'], row['Days since last coupon payment'], row['Coupon']), axis= 1)
    
    # sort the bonds by years until maturity
    jan6_sorted_by_years_until_maturity = jan6.sort_values(by='Years until Maturity', ascending=True)
    # print(jan6_sorted_by_years_until_maturity[['Dirty price', 'Years until Maturity']])

    return jan6_sorted_by_years_until_maturity

def calculate_all_yield_rates(bonds):
    
    # calcualte the yield rates
    yield_rates = {}
    for index, row in bonds.iterrows():
        t = row['Years until Maturity']
        if row['Dirty price'] == "-":
            continue
        if t < 0.5:
            yield_rates[t] = calculate_yield_rates_for_zero_coupon_bonds(row['Dirty price'], row['Coupon'], t)
        else:
            yield_rates[t] = calcualte_yield_rates(yield_rates, row['Dirty price'], row['Coupon'], t)
        print(f"t: {t}, yield rate: {yield_rates[t]}, dirty price: {row['Dirty price']}, coupon rate: {row['Coupon']}")
    
    for key in yield_rates:
        print(f"y: {key}, val: {yield_rates[key]}")

    years_until_maturity = list(yield_rates.keys())
    yield_rates_values = list(yield_rates.values())

    plt.plot(years_until_maturity, yield_rates_values, marker='o', linestyle='-', color='b')
    plt.show()

sorted_bonds = create_sorted_bonds_by_maturity()
calculate_all_yield_rates(sorted_bonds)