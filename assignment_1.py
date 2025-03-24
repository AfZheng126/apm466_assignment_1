import pandas as pd
import numpy as np

from datetime import datetime
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

def create_sorted_bonds_by_maturity(file_name):
    # If you want to read only a subset of sheets
    # xls = pd.ExcelFile('bond_prices.xlsx')
    # sheet_names = xls.sheet_names
    # Filter sheet names that start with 'Desired'
    # desired_sheets = [sheet for sheet in sheet_names if sheet.startswith('Desired')]
    # dfs = pd.read_excel('bond_prices.xlsx', sheet_name=desired_sheets, usecols = ['Coupon', 'Maturity Date', 'Bid', 'Ask', 'Years until Maturity'])

    dfs = pd.read_excel(file_name, sheet_name=None, usecols = ['Coupon', 'Maturity Date', 'Price', 'Years until Maturity'])
    
    for sheet_name, df in dfs.items():
        print(f"Processing sheet: {sheet_name}")
        # get all the required information for the bonds
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
        df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
        df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Price'], row['Days since last coupon payment'], row['Coupon']), axis= 1)
        df['Years until Maturity'] = df['Years until Maturity'].round(4)
        # print(df[['Dirty price', 'Years until Maturity']])

    return dfs

def read_extra_data(file_name):
    dfs = pd.read_excel(file_name, sheet_name=None, usecols = [''])

def calculate_days_since_last_coupon_payment(date):
    if date.month < 3: # since all coupons mature on March 1 or Sept 1
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

def calculate_all_spot_rates(bonds):
    all_spot_rates = {}
    for date, bond in bonds.items():
        # print(f"Processing sheet: {date}")
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
        
        # for key in yield_rates:
        #     print(f"y: {key}, val: {yield_rates[key]}")
        all_spot_rates[date] = spot_rates
    return all_spot_rates

def plot_spot_curve(spot_rates):
    cmap = plt.get_cmap('tab10')

    for i, date in enumerate(spot_rates):
        # plot curve for each date
        years_until_maturity = list(spot_rates[date].keys())
        spot_rates_values = list(spot_rates[date].values())
        
        # get the label for the plot
        words = date.split()
        name = ' '.join(words[1:])

        plt.plot(years_until_maturity, spot_rates_values, label = name, marker='o', linestyle='-', color=cmap(i / len(spot_rates)))
    
    # add labels and title
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Years until Maturity', fontsize=30)
    plt.ylabel('Yield', fontsize=30)
    plt.title('0-5 Year Spot Curve', fontsize=30)
    plt.legend(loc='lower right', fontsize=20)

def calculate_ytm(dirty_price, coupon_rate, years_until_maturity):
    coupon_payment = 100 * coupon_rate / 2
    if years_until_maturity < 0.5:
        notional = 100 + coupon_payment
        val = - np.log(dirty_price / notional) / years_until_maturity
        return val
    
    number_of_coupon_payments = int(np.floor(years_until_maturity * 2))

    # define function for continuous compounding ytm
    def equation(r):
        # consider all coupon payments
        present_value = 0
        for i in range(number_of_coupon_payments):
            t = years_until_maturity - (number_of_coupon_payments - i) * 0.5
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

# calculate the future price of a zero coupon bond
def calculate_future_price_zero_bonds(future_time, years_until_maturity, spot_rates, coupon_rate):
    if years_until_maturity in spot_rates:
            r = spot_rates[years_until_maturity]
    else:
        r = interpolate_rate(spot_rates, years_until_maturity)
    future_price = np.exp( - r * (years_until_maturity - future_time)) # no need for the 100 as it gets cancelled out in the log anyways
    return future_price

# create forward rate
def calculate_forward_rate(all_spot_rates, all_bonds):
    # iterate through the dates
    all_forward_rates = {}        

    for date in all_spot_rates:
        spot_rates = all_spot_rates[date]
        bonds = all_bonds[date]
        future_prices = []
        maturities = []
        
        # sort the bonds in increasing years until maturity
        bonds = bonds.sort_values(by='Years until Maturity', ascending=True)

        # calculate the value of the bonds at time t=1
        # iterate through all the different maturity dates between 1 and 4
        for index, row in bonds.iterrows():
            years_until_maturity = row['Years until Maturity']
            if row['Dirty price'] == "-" or years_until_maturity < 1 or (years_until_maturity in maturities):
                continue
            maturities.append(years_until_maturity)
            # future_prices.append(np.log(calculate_future_price(1, years_until_maturity, spot_rates, row['Coupon'])))
            future_prices.append(np.log(calculate_future_price_zero_bonds(1, years_until_maturity, spot_rates, row['Coupon'])))
        
        # print(f'maturities: {maturities}')
        # print(f'prices: {future_prices}')
        
        # now approximate the derivative using second order finite difference scheme for interior points and first order for boundary points
        forward_rates = {}
        number_of_points = len(future_prices)
        for index in range(number_of_points):
            if index == 0:
                forward_rates[maturities[index]] = - (future_prices[index + 1] - future_prices[index]) / (maturities[index + 1] - maturities[index])
                # print(f'val = {(future_prices[index + 1] - future_prices[index]) / (maturities[index + 1] - maturities[index])}')
            elif index == number_of_points - 1:
                forward_rates[maturities[index]] = - (future_prices[index] - future_prices[index - 1]) / (maturities[index] - maturities[index - 1])
            else: # take average of slope on both sides
                forward_rates[maturities[index]] = - 0.5 * ((future_prices[index + 1] - future_prices[index]) / (maturities[index + 1] - maturities[index]) + (future_prices[index] - future_prices[index - 1]) / (maturities[index] - maturities[index - 1]))
        
        all_forward_rates[date] = forward_rates

    return all_forward_rates

# calculate covariance matricies
def calculate_covariance_matricies_for_yields(all_rates, num_variables):
    # create the 5 random vectors X_i, 
    dates = list(all_rates.keys())
    num_rows = len(dates) - 1

    vectors = []
    for i in range(num_variables):
        vectors.append(np.zeros((num_rows, 1)))

    # vector for yield rates
    for j in range(num_rows): # each row is a date in our data collection period
        today_rates = all_rates[dates[j]]
        tomorrow_rates = all_rates[dates[j + 1]]
        for i in range(num_variables): 
            years_until_maturity = float(i + 1) 
            # the 5 variables are the 1, 2, 3, 4, 5 year rates for ytm
            # for forward rates it is the 1-1, 1-2, 1-3, 1-4, 1-5 year rates

            if years_until_maturity in today_rates:
                today_rate = today_rates[years_until_maturity]
            else:
                today_rate = interpolate_rate(today_rates, years_until_maturity)
            if years_until_maturity in tomorrow_rates:
                tomorrow_rate = tomorrow_rates[years_until_maturity]
            else:
                tomorrow_rate = interpolate_rate(tomorrow_rates, years_until_maturity)
            vectors[i][j] = np.log(tomorrow_rate / today_rate)

    # create the covariance matrix
    data = np.hstack(vectors)
    covariance_matrix = np.cov(data, rowvar=False)
    print(f'Covariance matrix: {covariance_matrix}')

    # calculate the Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

def calculate_covariance_matricies_for_forward_rates(all_rates, num_variables):
    # create the 5 random vectors X_i, 
    dates = list(all_rates.keys())
    num_rows = len(dates) - 1

    vectors = []
    for i in range(num_variables):
        vectors.append(np.zeros((num_rows, 1)))

    # vector for yield rates
    for j in range(num_rows): # each row is a date in our data collection period
        today_rates = all_rates[dates[j]]
        tomorrow_rates = all_rates[dates[j + 1]]
        for i in range(num_variables): 
            years_until_maturity = float(i + 1) 
            # for forward rates it is the 1-1, 1-2, 1-3, 1-4, 1-5 year rates

            if years_until_maturity in today_rates:
                today_rate = today_rates[years_until_maturity]
            else:
                # if years_until_maturity is 1, then there is no lower bound so just set it to the smallest year in today_rates
                if years_until_maturity == 1:
                    smallest_key = min(today_rates)
                    today_rate = today_rates[smallest_key]
                else:
                    today_rate = interpolate_rate(today_rates, years_until_maturity)
            if years_until_maturity in tomorrow_rates:
                tomorrow_rate = tomorrow_rates[years_until_maturity]
            else:
                if years_until_maturity == 1:
                    smallest_key = min(tomorrow_rates)
                    # print(f"smallest key: {smallest_key}")
                    tomorrow_rate = tomorrow_rates[smallest_key]
                else:
                    tomorrow_rate = interpolate_rate(tomorrow_rates, years_until_maturity)
            # print(f"date: {years_until_maturity}, today rate: {today_rate}, tomorrow rate: {tomorrow_rate}, value: {np.log(tomorrow_rate / today_rate)}")
            vectors[i][j] = np.log(tomorrow_rate / today_rate)

    # print(vectors)

    # create the covariance matrix
    data = np.hstack(vectors)
    covariance_matrix = np.cov(data, rowvar=False)
    print(f'Covariance matrix: {covariance_matrix}')

    # calculate the Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

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
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Years until Maturity', fontsize=30)
    plt.ylabel('YTM Value', fontsize=30)
    plt.title('0-5 Year YTM Curve', fontsize=30)
    plt.legend(loc='lower right', fontsize=20)

# plot forward rates
def plot_forward_rates_curve(forward_rates):

    cmap = plt.get_cmap('tab10')

    for i, date in enumerate(forward_rates):
        # plot curve for each date
        years_until_maturity = list(forward_rates[date].keys())
        forward_rates_values = list(forward_rates[date].values())
        
        # get the label for the plot
        words = date.split()
        name = ' '.join(words[1:])

        plt.plot(years_until_maturity, forward_rates_values, label = name, marker='o', linestyle='-', color=cmap(i / len(forward_rates)))
    
    # add labels and title
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('T = Years until Maturity', fontsize=30)
    plt.ylabel('f(1, T)', fontsize=30)
    plt.title('Forward Rate Curve', fontsize=30)
    plt.legend(loc='lower right', fontsize=20)

# get the data
sorted_bonds = create_sorted_bonds_by_maturity('bond_prices_chosen_bonds.xlsx')
spot_rates = calculate_all_spot_rates(sorted_bonds)
yield_rates = calculate_ytm_curve(sorted_bonds)
forward_rates = calculate_forward_rate(spot_rates, sorted_bonds)

# print(forward_rates)

# get the eigenvalues
print("covariance for ytm rates")
calculate_covariance_matricies_for_yields(yield_rates, 5)

print("covariance for forward rates")
calculate_covariance_matricies_for_forward_rates(forward_rates, 5)


# plot the data
plt.figure()
plot_spot_curve(spot_rates)
plt.figure()
plot_ytm_curve(yield_rates)
plt.figure()
plot_forward_rates_curve(forward_rates)

plt.show()
